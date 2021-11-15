# Copyright 2021 Bouazza SAADEDDINE

# This file is part of NeuralXVA.

# NeuralXVA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# NeuralXVA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with NeuralXVA.  If not, see <https://www.gnu.org/licenses/>.

import math
import numba as nb
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_dtype


def compile_cuda_generate_exp1(num_spreads, num_defs_per_path, num_paths, ntpb, stream):

    num_names = num_spreads - 1

    sig = (nb.float32[:, :, :], nb.from_dtype(xoroshiro128p_dtype)[:])
    
    @cuda.jit(func_or_sig=sig, max_registers=64)
    def _cuda_generate_exp1(out, rng_states):
        block_x = cuda.blockIdx.x
        block_y = cuda.blockIdx.y
        block_size = cuda.blockDim.x
        tidx = cuda.threadIdx.x
        pos = tidx + block_y * block_size
        if pos < num_paths:
            for i in range(num_names):
                out[i, block_x, pos] = -math.log(xoroshiro128p_uniform_float32(rng_states, block_x*num_paths+pos))

    _cuda_generate_exp1._func.get().cache_config(prefer_cache=True)
    cuda_generate_exp1 = _cuda_generate_exp1[(num_defs_per_path, (num_paths+ntpb-1)//ntpb), ntpb, stream]

    # returning the compiled kernel
    return cuda_generate_exp1


def compile_cuda_bulk_diffuse(g_diff_params, g_L_T, num_fine_per_coarse,
                              num_rates, num_spreads, num_defs_per_path, num_paths, ntpb,
                              stream):
    # compile-time constants
    num_diffusions = 2*num_rates+num_spreads-1
    fx_start = num_rates
    fx_params_start = 3*num_rates
    drift_adj_start = 4*num_rates - 1
    spread_start = fx_start + num_rates - 1
    spread_params_start = fx_params_start + 2*num_rates - 2

    sig = (nb.int32, nb.float32, nb.float32[:, :, :], nb.int8[:, :, :, :], nb.float32[:, :], nb.float32[:, :, :],
           nb.float32[:, :], nb.int32[:, :], nb.float32[:, :, :], nb.from_dtype(xoroshiro128p_dtype)[:], nb.float32, nb.int32)

    @cuda.jit(func_or_sig=sig, max_registers=64)
    def _cuda_bulk_diffuse(coarse_idx, t, X, def_indicators, dom_rate_integral, spread_integrals, irs_f32, irs_i32, exp_1, rng_states, dt, max_coarse_per_reset):
        block = cuda.blockIdx.x
        block_size = cuda.blockDim.x
        tidx = cuda.threadIdx.x
        pos = tidx + block * block_size

        if pos < num_paths:
            diff_params = cuda.const.array_like(g_diff_params)
            L_T = cuda.const.array_like(g_L_T)
            dW_corr = cuda.local.array(num_diffusions, nb.float32)
            tmp_X = cuda.local.array(num_diffusions, nb.float32)
            tmp_spread_step_integrals = cuda.local.array(num_spreads, nb.float32)
            tmp_dom_rate_step_integral = nb.float32(0)

            for i in range(num_diffusions):
                tmp_X[i] = X[coarse_idx+max_coarse_per_reset-2, i, pos]

            for i in range(num_spreads):
                tmp_spread_step_integrals[i] = 0

            sqrt_dt = math.sqrt(dt)

            for i in range(num_rates-1):
                tmp_X[fx_start+i] = math.log(tmp_X[fx_start+i])

            for fine_idx in range(num_fine_per_coarse):
                for i in range(num_diffusions):
                    dW_corr[i] = 0

                for i in range(num_diffusions):
                    u = xoroshiro128p_uniform_float32(rng_states, pos)
                    v = xoroshiro128p_uniform_float32(rng_states, pos)
                    v = math.sqrt(-2*math.log(u)) * math.cos(2*math.pi*v) * sqrt_dt # Box-Muller, throwing the other normal away
                    for j in range(i, num_diffusions):
                        # L_T is the transpose of the lower-triangular L such that Corr=L*L_T
                        dW_corr[j] += L_T[i*num_diffusions-i*(i+1)//2+j] * v
                # E[Au*(Au)^T] = E[A*u*u^T*A^T] = A*Cov*A^T -> for unit Cov, it is enough to choose A=L
                # E[LdW*(LdW)^T] = dt*LL^T = dt*Corr
                # dW_corr[k] = sum_j L_{i,j} * dW_j

                # FX log-diffusions
                for i in range(num_rates-1):
                    tmp_X[fx_start+i] += (tmp_X[0] - tmp_X[i+1] - 0.5*diff_params[fx_params_start+i]** 2) * dt + diff_params[fx_params_start+i] * dW_corr[fx_start+i]

                # rate diffusions
                # TODO: change this and diffuse jointly the short rate and its integral exactly
                # (but for now let's just stick with a numerical integral)
                tmp_dom_rate_step_integral += 0.5 * tmp_X[0] * dt

                for i in range(num_rates):
                    tmp_X[i] += diff_params[i] * \
                        (diff_params[num_rates+i] - tmp_X[i]) * dt
                    drift_adj = 0.
                    if i != 0:
                        drift_adj = diff_params[drift_adj_start+i-1]
                    tmp_X[i] += diff_params[2*num_rates+i] * \
                        (dW_corr[i] + drift_adj * dt)

                tmp_dom_rate_step_integral += 0.5 * tmp_X[0] * dt

                # spread diffusions
                for i in range(num_spreads):
                    pos_spread = max(tmp_X[spread_start+i], 0.)
                    tmp_X[spread_start+i] += diff_params[spread_params_start+i] * \
                        (diff_params[spread_params_start +
                                     num_spreads+i] - pos_spread) * dt
                    tmp_X[spread_start+i] += diff_params[spread_params_start+2 *
                                                         num_spreads+i] * math.sqrt(pos_spread) * dW_corr[spread_start+i]
                    tmp_spread_step_integrals[i] += 0.5 * pos_spread * dt
                    if tmp_X[spread_start+i] > 0.:
                        tmp_spread_step_integrals[i] += 0.5 * \
                            tmp_X[spread_start+i] * dt

            for i in range(num_diffusions):
                if num_rates <= i < 2*num_rates-1:
                    X[coarse_idx+max_coarse_per_reset-1, i, pos] = math.exp(tmp_X[i])
                else:
                    X[coarse_idx+max_coarse_per_reset-1, i, pos] = tmp_X[i]

            for i in range(num_spreads):
                spread_integrals[coarse_idx, i, pos] = spread_integrals[coarse_idx - 1, i, pos] + tmp_spread_step_integrals[i]

            dom_rate_integral[coarse_idx, pos] = dom_rate_integral[coarse_idx - 1, pos] + tmp_dom_rate_step_integral

            for i in range(num_spreads-1):
                s = spread_integrals[coarse_idx, i+1, pos]
                q = i // 8
                r = i % 8
                for j in range(num_defs_per_path):
                    if s > exp_1[i, j, pos]:
                        # no common-shock this time
                        def_indicators[coarse_idx, q, j, pos] |= (1 << r)
            
    _cuda_bulk_diffuse._func.get().cache_config(prefer_cache=True)
    cuda_bulk_diffuse = _cuda_bulk_diffuse[(num_paths+ntpb-1)//ntpb, ntpb, stream]
    
    # finally, return the compiled kernel
    return cuda_bulk_diffuse


def compile_cuda_diffuse_and_price(irs_batch_size, vanilla_batch_size, g_diff_params, g_R, g_L_T, num_fine_per_coarse, num_rates, num_spreads, num_paths, ntpb, stream):
    # compile-time constants
    num_cpty = num_spreads - 1
    num_diffusions = 2*num_rates+num_spreads-1
    fx_start = num_rates
    fx_params_start = 3*num_rates
    drift_adj_start = 4*num_rates - 1
    spread_start = fx_start + num_rates - 1
    spread_params_start = fx_params_start + 2*num_rates - 2

    sig = (nb.int32, nb.int32, nb.float32, nb.float32[:, :, :], nb.float32[:, :], nb.float32[:, :, :], nb.float32[:, :, :], nb.float32[:, :, :], nb.float32[:, :], nb.int32[:, :], nb.float32[:, :], nb.int32[:, :], nb.bool_[:, :], nb.from_dtype(xoroshiro128p_dtype)[:], nb.float32, nb.int32, nb.int32)

    @cuda.jit(func_or_sig=sig, max_registers=64)
    def _cuda_bulk_diffuse_and_price(coarse_start_idx, num_coarse_steps, t, X, dom_rate_integral, spread_integrals, mtm_by_cpty, cash_flows_by_cpty, irs_f32, irs_i32, vanillas_on_fx_f32, vanillas_on_fx_i32, vanillas_on_fx_b8, rng_states, dt, max_coarse_per_reset, window_length):
        block = cuda.blockIdx.x
        block_size = cuda.blockDim.x
        tidx = cuda.threadIdx.x
        pos = tidx + block * block_size

        if pos < num_paths:
            diff_params = cuda.const.array_like(g_diff_params)
            # TODO: reuse same arrays for product cache...
            irs_f32_sh = cuda.shared.array(shape=(irs_batch_size, 4), dtype=nb.float32)
            irs_i32_sh = cuda.shared.array(shape=(irs_batch_size, 3), dtype=nb.int32)
            vanillas_on_fx_f32_sh = cuda.shared.array(shape=(vanilla_batch_size, 3), dtype=nb.float32)
            vanillas_on_fx_i32_sh = cuda.shared.array(shape=(vanilla_batch_size, 2), dtype=nb.int32)
            vanillas_on_fx_b8_sh = cuda.shared.array(shape=(vanilla_batch_size, 1), dtype=nb.bool_)
            R = cuda.const.array_like(g_R)
            L_T = cuda.const.array_like(g_L_T)
            dW_corr = cuda.local.array(num_diffusions, nb.float32)
            tmp_X = cuda.local.array(num_diffusions, nb.float32)
            tmp_spread_integrals = cuda.local.array(num_spreads, nb.float32)
            tmp_mtm_by_cpty = cuda.local.array(num_cpty, nb.float32)
            tmp_cash_flows_by_cpty = cuda.local.array(num_cpty, nb.float32)

            sqrt_dt = math.sqrt(dt)

            for i in range(num_diffusions):
                tmp_X[i] = X[coarse_start_idx+max_coarse_per_reset-2, i, pos]
            
            tmp_dom_rate_integral = dom_rate_integral[coarse_start_idx - 1, pos]

            for i in range(num_spreads):
                tmp_spread_integrals[i] = spread_integrals[coarse_start_idx - 1, i, pos]

            for coarse_idx in range(coarse_start_idx, coarse_start_idx+num_coarse_steps):
                for i in range(num_rates-1):
                    tmp_X[fx_start+i] = math.log(tmp_X[fx_start+i])

                for fine_idx in range(num_fine_per_coarse):
                    for i in range(num_diffusions):
                        dW_corr[i] = 0

                    for i in range(num_diffusions):
                        u = xoroshiro128p_uniform_float32(rng_states, pos)
                        v = xoroshiro128p_uniform_float32(rng_states, pos)
                        v = math.sqrt(-2*math.log(u)) * math.cos(2*math.pi*v) * sqrt_dt # Box-Muller, throwing the other normal away
                        for j in range(i, num_diffusions):
                            # L_T is the transpose of the lower-triangular L such that Corr=L*L_T
                            dW_corr[j] += L_T[i*num_diffusions-i*(i+1)//2+j] * v
                    # E[Au*(Au)^T] = E[A*u*u^T*A^T] = A*Cov*A^T -> for unit Cov, it is enough to choose A=L
                    # E[LdW*(LdW)^T] = dt*LL^T = dt*Corr
                    # dW_corr[k] = sum_j L_{i,j} * dW_j

                    # FX log-diffusions
                    for i in range(num_rates-1):
                        tmp_X[fx_start+i] += (tmp_X[0] - tmp_X[i+1] - 0.5*diff_params[fx_params_start+i]** 2) * dt + diff_params[fx_params_start+i] * dW_corr[fx_start+i]

                    # rate diffusions
                    # TODO: change this and diffuse jointly the short rate and its integral exactly
                    # (but for now let's just stick with a numerical integral)
                    tmp_dom_rate_integral += 0.5 * tmp_X[0] * dt

                    for i in range(num_rates):
                        tmp_X[i] += diff_params[i] * \
                            (diff_params[num_rates+i] - tmp_X[i]) * dt
                        drift_adj = nb.float32(0)
                        if i != 0:
                            drift_adj = diff_params[drift_adj_start+i-1]
                        tmp_X[i] += diff_params[2*num_rates+i] * (dW_corr[i] + drift_adj * dt)

                    tmp_dom_rate_integral += 0.5 * tmp_X[0] * dt

                    # spread diffusions
                    for i in range(num_spreads):
                        pos_spread = max(tmp_X[spread_start+i], 0)
                        tmp_X[spread_start+i] += diff_params[spread_params_start+i] * (diff_params[spread_params_start + num_spreads+i] - pos_spread) * dt
                        tmp_X[spread_start+i] += diff_params[spread_params_start+2*num_spreads+i] * math.sqrt(pos_spread) * dW_corr[spread_start+i]
                        tmp_spread_integrals[i] += 0.5 * pos_spread * dt
                        if tmp_X[spread_start+i] > 0:
                            tmp_spread_integrals[i] += 0.5 * tmp_X[spread_start+i] * dt

                for i in range(num_rates-1):
                    tmp_X[fx_start+i] = math.exp(tmp_X[fx_start+i])
                
                if pos==0:
                    print('[ OUTER | rate 0 | t =', t, '] r =', tmp_X[0], '| sliding_window = (', X[coarse_idx-1+max_coarse_per_reset-2, 0, 0], '|', X[coarse_idx-1+max_coarse_per_reset-1, 0, 0], ')')
                
                for cpty in range(num_cpty):
                    tmp_mtm_by_cpty[cpty] = 0
                    tmp_cash_flows_by_cpty[cpty] = 0
                
                for batch_idx in range((vanillas_on_fx_f32.shape[0]+vanilla_batch_size-1)//vanilla_batch_size):
                    cuda.syncthreads()
                    if tidx == 0:
                        for i in range(vanilla_batch_size):
                            if batch_idx*vanilla_batch_size+i < vanillas_on_fx_f32.shape[0]:
                                for j in range(vanillas_on_fx_f32.shape[1]):
                                    vanillas_on_fx_f32_sh[i, j] = vanillas_on_fx_f32[batch_idx*vanilla_batch_size+i, j]
                                for j in range(vanillas_on_fx_i32.shape[1]):
                                    vanillas_on_fx_i32_sh[i, j] = vanillas_on_fx_i32[batch_idx*vanilla_batch_size+i, j]
                                for j in range(vanillas_on_fx_b8.shape[1]):
                                    vanillas_on_fx_b8_sh[i, j] = vanillas_on_fx_b8[batch_idx*vanilla_batch_size+i, j]
                            else:
                                i -= 1
                                break
                    else:
                        i = min(vanillas_on_fx_f32.shape[0]-batch_idx*vanilla_batch_size, vanilla_batch_size)-1
                    cuda.syncthreads()
                    for j in range(i+1):
                        maturity = vanillas_on_fx_f32_sh[j, 0]
                        if maturity + 0.1 * dt < t:
                            continue
                        notional = vanillas_on_fx_f32_sh[j, 1]
                        strike = vanillas_on_fx_f32_sh[j, 2]
                        cpty = vanillas_on_fx_i32_sh[j, 0]
                        undl = vanillas_on_fx_i32_sh[j, 1]
                        call_put = vanillas_on_fx_b8_sh[j, 0]
                        a_d = diff_params[0]
                        a_f = diff_params[undl]
                        b_d = diff_params[num_rates]
                        b_f = diff_params[num_rates+undl]
                        s_d = diff_params[2*num_rates]
                        s_f = diff_params[2*num_rates+undl]
                        s_fx = diff_params[3*num_rates+undl-1]
                        price = _cuda_price_vanilla_on_fx(call_put, strike, t, maturity, tmp_X[0], tmp_X[undl],
                                                        tmp_X[num_rates+undl-1], R[num_rates+undl-1],
                                                        R[undl*num_diffusions-undl*(undl+1)//2+num_rates+undl-1],
                                                        R[undl], a_d, a_f, b_d, b_f, s_d, s_f, s_fx, dt)
                        if coarse_idx == coarse_start_idx + 1 and pos == 511:
                            print('* coarse_idx =', coarse_idx, ', global_thread =', pos,'| maturity =', maturity, '| strike =', strike, '| price =', price, '| fx spot =', tmp_X[num_rates+undl-1], '| fx vol = ', s_fx, '| cpty = ', cpty, '| undl = ', undl)
                        for _cpty in range(num_cpty):
                            tmp_mtm_by_cpty[_cpty] += notional * price * (_cpty == cpty)
                        if t > maturity - 0.1*dt:
                            if call_put:
                                payoff = tmp_X[num_rates+undl-1] - strike
                            else:
                                payoff = strike - tmp_X[num_rates+undl-1]
                            if payoff < 0:
                                payoff = 0
                            for _cpty in range(num_cpty):
                                tmp_cash_flows_by_cpty[_cpty] += notional * payoff * (_cpty == cpty)
                
                for batch_idx in range((irs_f32.shape[0]+irs_batch_size-1)//irs_batch_size):
                    cuda.syncthreads()
                    if tidx == 0:
                        for i in range(irs_batch_size):
                            if batch_idx*irs_batch_size+i < irs_f32.shape[0]:
                                for j in range(irs_f32.shape[1]):
                                    irs_f32_sh[i, j] = irs_f32[batch_idx*irs_batch_size+i, j]
                                for j in range(irs_i32.shape[1]):
                                    irs_i32_sh[i, j] = irs_i32[batch_idx*irs_batch_size+i, j]
                            else:
                                i -= 1
                                break
                    else:
                        i = min(irs_f32.shape[0]-batch_idx*irs_batch_size, irs_batch_size)-1
                    cuda.syncthreads()
                    for j in range(i+1):
                        first_reset = irs_f32_sh[j, 0]
                        reset_freq = irs_f32_sh[j, 1]
                        num_resets = irs_i32_sh[j, 0]
                        if first_reset + (num_resets - 1) * reset_freq + 0.1 * dt < t:
                            continue
                        notional = irs_f32_sh[j, 2]
                        cpty = irs_i32_sh[j, 1]
                        ccy = irs_i32_sh[j, 2]
                        fx = nb.float32(1)
                        if ccy != 0:
                            fx = tmp_X[num_rates + ccy - 1]
                        a = diff_params[ccy]
                        b = diff_params[num_rates+ccy]
                        sigma = diff_params[2*num_rates+ccy]
                        swap_rate = irs_f32_sh[j, 3]
                        if t > first_reset - 0.1*dt:
                            num_coarse_per_reset = int((reset_freq+dt)/(num_fine_per_coarse*dt))
                            m = int((t - first_reset - (num_fine_per_coarse-1)*dt) / reset_freq) # locate the strictly previous reset date in the resets grid
                            m = int((t-first_reset-m*reset_freq+dt)/(num_fine_per_coarse*dt)) # locate it now in the coarse grid
                        else:
                            m = nb.int32(1)
                        price = _cuda_price_irs(ccy, swap_rate, X[coarse_idx-m+max_coarse_per_reset-1, ccy, pos], tmp_X[ccy], t, first_reset, reset_freq, num_resets, False, a, b, sigma, dt)
                        for _cpty in range(num_cpty):
                            tmp_mtm_by_cpty[_cpty] += notional * fx * price * (_cpty == cpty)
                        k = int((t-first_reset+0.1*dt)/reset_freq)
                        is_coupon_date = (k >= 1) and (abs(t-first_reset-k*reset_freq) < 0.1*dt)
                        if is_coupon_date:
                            for _cpty in range(num_cpty):
                                tmp_cash_flows_by_cpty[_cpty] += notional * fx * (_cuda_price_zc_bond_inv(ccy, X[coarse_idx-m+max_coarse_per_reset-1, ccy, pos], 0, reset_freq, a, b, sigma) - 1 - swap_rate * reset_freq) * (_cpty == cpty)
                
                for i in range(num_diffusions):
                    X[coarse_idx+max_coarse_per_reset-1, i, pos] = tmp_X[i]

                for i in range(num_spreads):
                    spread_integrals[coarse_idx, i, pos] = tmp_spread_integrals[i]

                dom_rate_integral[coarse_idx, pos] = tmp_dom_rate_integral
                
                for cpty in range(num_cpty):
                    mtm_by_cpty[coarse_idx, cpty, pos] = tmp_mtm_by_cpty[cpty]
                    cash_flows_by_cpty[coarse_idx, cpty, pos] = tmp_cash_flows_by_cpty[cpty]
            
                t += dt * num_fine_per_coarse
            
    cuda_bulk_diffuse_and_price = _cuda_bulk_diffuse_and_price[(num_paths+ntpb-1)//ntpb, ntpb, stream]
    
    # finally, return the compiled kernel
    return cuda_bulk_diffuse_and_price

def compile_cuda_oversimulate_defs(num_spreads, num_defs_per_path, num_paths, ntpb, stream):
    # compile-time constants
    num_cpty = num_spreads - 1

    sig = (nb.int32, nb.int32, nb.int8[:, :, :, :], nb.float32[:, :, :], nb.float32[:, :, :])

    @cuda.jit(func_or_sig=sig, max_registers=64)
    def _cuda_oversimulate_defs(coarse_start_idx, num_coarse_steps, def_indicators, spread_integrals, exp_1):
        block_x = cuda.blockIdx.x
        block_y = cuda.blockIdx.y
        block_size = cuda.blockDim.x
        tidx = cuda.threadIdx.x
        pos = tidx + block_y * block_size

        if pos < num_paths:
            for coarse_idx in range(coarse_start_idx, coarse_start_idx+num_coarse_steps):
                for i in range(num_cpty):
                    s = spread_integrals[coarse_idx, i+1, pos]
                    q = i // 8
                    r = i % 8
                    if s > exp_1[i, block_x, pos]:
                        # no common-shock this time
                        def_indicators[coarse_idx, q, block_x, pos] |= (1 << r)
            
    cuda_oversimulate_defs = _cuda_oversimulate_defs[(num_defs_per_path, (num_paths+ntpb-1)//ntpb), ntpb, stream]
    
    # finally, return the compiled kernel
    return cuda_oversimulate_defs

def compile_cuda_nested_cva(irs_batch_size, vanilla_batch_size, g_diff_params, g_R, g_L_T, num_fine_per_coarse, num_rates, num_spreads, num_defs_per_path, num_paths, num_inner_paths, max_coarse_per_reset, stream):
    # compile-time constants
    num_cpty = num_spreads - 1
    num_cpty_buckets = (num_cpty+7)//8
    num_diffusions = 2*num_rates+num_spreads-1
    fx_start = num_rates
    fx_params_start = 3*num_rates
    drift_adj_start = 4*num_rates - 1
    spread_start = fx_start + num_rates - 1
    spread_params_start = fx_params_start + 2*num_rates - 2
    full_mask = nb.uint32(-1)

    if num_inner_paths & (num_inner_paths-1) == 0:
        inner_stride = num_inner_paths
    else:
        c = 0
        d = num_inner_paths
        while d != 0:
            d >>= 1
            c += 1
        inner_stride = 1 << c

    sig = (nb.int32, nb.int32, nb.float32, nb.float32[:, :, :], nb.int8[:, :, :, :], nb.float32[:, :], nb.float32[:, :, :], nb.float32[:, :, :], nb.float32[:, :, :], nb.float32[:, :], nb.int32[:, :], nb.float32[:, :], nb.int32[:, :], nb.bool_[:, :], nb.float32[:, :, :], nb.from_dtype(xoroshiro128p_dtype)[:], nb.float32, nb.int32, nb.bool_, nb.float32[:, :], nb.float32[:, :])

    @cuda.jit(func_or_sig=sig, max_registers=64)
    def _cuda_nested_cva(coarse_start_idx, num_coarse_steps, t, X, def_indicators, dom_rate_integral, spread_integrals, mtm_by_cpty, cash_flows_by_cpty, irs_f32, irs_i32, vanillas_on_fx_f32, vanillas_on_fx_i32, vanillas_on_fx_b8, exp_1, rng_states, dt, window_length, indicator_in_cva, out1, out2):
        block = cuda.blockIdx.x
        block_size = cuda.blockDim.x
        tidx = cuda.threadIdx.x
        pos = tidx + block * block_size

        diff_params = cuda.const.array_like(g_diff_params)
        R = cuda.const.array_like(g_R)
        irs_f32_sh = cuda.shared.array(shape=(irs_batch_size, 4), dtype=nb.float32)
        irs_i32_sh = cuda.shared.array(shape=(irs_batch_size, 3), dtype=nb.int32)
        vanillas_on_fx_f32_sh = cuda.shared.array(shape=(vanilla_batch_size, 3), dtype=nb.float32)
        vanillas_on_fx_i32_sh = cuda.shared.array(shape=(vanilla_batch_size, 2), dtype=nb.int32)
        vanillas_on_fx_b8_sh = cuda.shared.array(shape=(vanilla_batch_size, 1), dtype=nb.bool_)
        L_T = cuda.const.array_like(g_L_T)
        dW_corr = cuda.local.array(num_diffusions, nb.float32)
        tmp_X = cuda.local.array(num_diffusions, nb.float32)
        tmp_exp_1 = cuda.local.array((num_cpty, num_defs_per_path), nb.float32)
        tmp_rates_sliding_window = cuda.local.array((max_coarse_per_reset, num_rates), nb.float32)
        tmp_spread_integrals_prev = cuda.local.array(num_spreads, nb.float32)
        tmp_spread_integrals = cuda.local.array(num_spreads, nb.float32)
        tmp_def_indicators = cuda.local.array((num_cpty_buckets, num_defs_per_path), nb.int8)
        tmp_mtm_by_cpty = cuda.local.array(num_cpty, nb.float32)
        tmp_cva_payoff_by_cpty = cuda.local.array((num_cpty, num_defs_per_path), nb.float32)
        cva_payoff_by_cpty_sh = cuda.shared.array(inner_stride, nb.float32)
        cva_payoff_by_cpty_sq_sh = cuda.shared.array(inner_stride, nb.float32)

        cva_payoff_by_cpty_sh[tidx] = 0
        cva_payoff_by_cpty_sq_sh[tidx] = 0
        cuda.syncthreads()

        if tidx < num_inner_paths:
            sqrt_dt = math.sqrt(dt)

            for i in range(num_cpty):
                for j in range(num_defs_per_path):
                    tmp_cva_payoff_by_cpty[i, j] = 0
                    tmp_exp_1[i, j] = -math.log(xoroshiro128p_uniform_float32(rng_states, num_paths*num_defs_per_path+pos)) # simulate exp1 here

            for i in range(num_diffusions):
                tmp_X[i] = X[coarse_start_idx+max_coarse_per_reset-1, i, block]
            
            for j in range(max_coarse_per_reset):
                for i in range(num_rates):
                    tmp_rates_sliding_window[max_coarse_per_reset-j-1, i] = X[coarse_start_idx+max_coarse_per_reset-2-j, i, block]
            
            tmp_dom_rate_integral = nb.float32(0)

            for i in range(num_spreads):
                tmp_spread_integrals[i] = 0
            
            for q in range(num_cpty_buckets):
                for j in range(num_defs_per_path):
                    tmp_def_indicators[q, j] = def_indicators[coarse_start_idx, q, j, block]

            for coarse_idx in range(coarse_start_idx, coarse_start_idx+num_coarse_steps):
                t += dt * num_fine_per_coarse
                for i in range(num_rates-1):
                    tmp_X[fx_start+i] = math.log(tmp_X[fx_start+i])
                for i in range(num_spreads):
                    tmp_spread_integrals_prev[i] = tmp_spread_integrals[i]

                for fine_idx in range(num_fine_per_coarse):
                    for i in range(num_diffusions):
                        dW_corr[i] = 0

                    for i in range(num_diffusions):
                        u = xoroshiro128p_uniform_float32(rng_states, num_paths*num_defs_per_path+pos)
                        v = xoroshiro128p_uniform_float32(rng_states, num_paths*num_defs_per_path+pos)
                        v = math.sqrt(-2*math.log(u)) * math.cos(2*math.pi*v) * sqrt_dt # Box-Muller, throwing the other normal away
                        for j in range(i, num_diffusions):
                            # L_T is the transpose of the lower-triangular L such that Corr=L*L_T
                            dW_corr[j] += L_T[i*num_diffusions-i*(i+1)//2+j] * v
                    # E[Au*(Au)^T] = E[A*u*u^T*A^T] = A*Cov*A^T -> for unit Cov, it is enough to choose A=L
                    # E[LdW*(LdW)^T] = dt*LL^T = dt*Corr
                    # dW_corr[k] = sum_j L_{i,j} * dW_j

                    # FX log-diffusions
                    for i in range(num_rates-1):
                        tmp_X[fx_start+i] += (tmp_X[0] - tmp_X[i+1] - 0.5*diff_params[fx_params_start+i]** 2) * dt + diff_params[fx_params_start+i] * dW_corr[fx_start+i]

                    # rate diffusions
                    # TODO: change this and diffuse jointly the short rate and its integral exactly
                    # (but for now let's just stick with a numerical integral)
                    tmp_dom_rate_integral += 0.5 * tmp_X[0] * dt

                    for i in range(num_rates):
                        tmp_X[i] += diff_params[i] * \
                            (diff_params[num_rates+i] - tmp_X[i]) * dt
                        drift_adj = nb.float32(0)
                        if i != 0:
                            drift_adj = diff_params[drift_adj_start+i-1]
                        tmp_X[i] += diff_params[2*num_rates+i] * (dW_corr[i] + drift_adj * dt)

                    tmp_dom_rate_integral += 0.5 * tmp_X[0] * dt

                    # spread diffusions
                    for i in range(num_spreads):
                        pos_spread = max(tmp_X[spread_start+i], 0)
                        tmp_X[spread_start+i] += diff_params[spread_params_start+i] * (diff_params[spread_params_start + num_spreads+i] - pos_spread) * dt
                        tmp_X[spread_start+i] += diff_params[spread_params_start+2*num_spreads+i] * math.sqrt(pos_spread) * dW_corr[spread_start+i]
                        tmp_spread_integrals[i] += 0.5 * pos_spread * dt
                        if tmp_X[spread_start+i] > 0:
                            tmp_spread_integrals[i] += 0.5 * tmp_X[spread_start+i] * dt

                for i in range(num_rates):
                    for j in range(max_coarse_per_reset-1):
                        tmp_rates_sliding_window[j, i] = tmp_rates_sliding_window[j+1, i]
                    tmp_rates_sliding_window[max_coarse_per_reset-1, i] = tmp_X[i]
                
                for i in range(num_rates-1):
                    tmp_X[fx_start+i] = math.exp(tmp_X[fx_start+i])
                
                for cpty in range(num_cpty):
                    tmp_mtm_by_cpty[cpty] = 0

                for batch_idx in range((vanillas_on_fx_f32.shape[0]+vanilla_batch_size-1)//vanilla_batch_size):
                    cuda.syncthreads()
                    if tidx == 0:
                        for i in range(vanilla_batch_size):
                            if batch_idx*vanilla_batch_size+i < vanillas_on_fx_f32.shape[0]:
                                for j in range(vanillas_on_fx_f32.shape[1]):
                                    vanillas_on_fx_f32_sh[i, j] = vanillas_on_fx_f32[batch_idx*vanilla_batch_size+i, j]
                                for j in range(vanillas_on_fx_i32.shape[1]):
                                    vanillas_on_fx_i32_sh[i, j] = vanillas_on_fx_i32[batch_idx*vanilla_batch_size+i, j]
                                for j in range(vanillas_on_fx_b8.shape[1]):
                                    vanillas_on_fx_b8_sh[i, j] = vanillas_on_fx_b8[batch_idx*vanilla_batch_size+i, j]
                            else:
                                i -= 1
                                break
                    else:
                        i = min(vanillas_on_fx_f32.shape[0]-batch_idx*vanilla_batch_size, vanilla_batch_size)-1
                    cuda.syncthreads()
                    for j in range(i+1):
                        maturity = vanillas_on_fx_f32_sh[j, 0]
                        if maturity + 0.1 * dt < t:
                            continue
                        notional = vanillas_on_fx_f32_sh[j, 1]
                        strike = vanillas_on_fx_f32_sh[j, 2]
                        cpty = vanillas_on_fx_i32_sh[j, 0]
                        undl = vanillas_on_fx_i32_sh[j, 1]
                        call_put = vanillas_on_fx_b8_sh[j, 0]
                        a_d = diff_params[0]
                        a_f = diff_params[undl]
                        b_d = diff_params[num_rates]
                        b_f = diff_params[num_rates+undl]
                        s_d = diff_params[2*num_rates]
                        s_f = diff_params[2*num_rates+undl]
                        s_fx = diff_params[3*num_rates+undl-1]
                        price = _cuda_price_vanilla_on_fx(call_put, strike, t, maturity, tmp_X[0], tmp_X[undl],
                                                        tmp_X[num_rates+undl-1], R[num_rates+undl-1],
                                                        R[undl*num_diffusions-undl*(undl+1)//2+num_rates+undl-1],
                                                        R[undl], a_d, a_f, b_d, b_f, s_d, s_f, s_fx, dt)
                        for _cpty in range(num_cpty):
                            tmp_mtm_by_cpty[_cpty] += notional * price * (_cpty == cpty)
                
                for batch_idx in range((irs_f32.shape[0]+irs_batch_size-1)//irs_batch_size):
                    cuda.syncthreads()
                    if tidx == 0:
                        for i in range(irs_batch_size):
                            if batch_idx*irs_batch_size+i < irs_f32.shape[0]:
                                for j in range(irs_f32.shape[1]):
                                    irs_f32_sh[i, j] = irs_f32[batch_idx*irs_batch_size+i, j]
                                for j in range(irs_i32.shape[1]):
                                    irs_i32_sh[i, j] = irs_i32[batch_idx*irs_batch_size+i, j]
                            else:
                                i -= 1
                                break
                    else:
                        i = min(irs_f32.shape[0]-batch_idx*irs_batch_size, irs_batch_size)-1
                    cuda.syncthreads()
                    for j in range(i+1):
                        first_reset = irs_f32_sh[j, 0]
                        reset_freq = irs_f32_sh[j, 1]
                        num_resets = irs_i32_sh[j, 0]
                        if first_reset + (num_resets - 1) * reset_freq + 0.1 * dt < t:
                            continue
                        notional = irs_f32_sh[j, 2]
                        cpty = irs_i32_sh[j, 1]
                        ccy = irs_i32_sh[j, 2]
                        fx = nb.float32(1)
                        if ccy != 0:
                            fx = tmp_X[num_rates + ccy - 1]
                        a = diff_params[ccy]
                        b = diff_params[num_rates+ccy]
                        sigma = diff_params[2*num_rates+ccy]
                        swap_rate = irs_f32_sh[j, 3]
                        if t > first_reset - 0.1*dt:
                            m = int((t - first_reset - (num_fine_per_coarse-1)*dt) / reset_freq) # locate the strictly previous reset date in the resets grid
                            m = int((t-first_reset-m*reset_freq+dt)/(num_fine_per_coarse*dt)) # locate it now in the coarse grid
                            m = nb.int32(max_coarse_per_reset-m)
                        else:
                            m = nb.int32(max_coarse_per_reset-1)
                        price = _cuda_price_irs(ccy, swap_rate, tmp_rates_sliding_window[m, ccy], tmp_X[ccy], t, first_reset, reset_freq, num_resets, False, a, b, sigma, dt)
                        tmp_mtm_by_cpty[cpty] += notional * fx * price
                
                discount_factor = math.exp(-tmp_dom_rate_integral)
                
                for i in range(num_cpty):
                    s = tmp_spread_integrals[i+1]
                    q = i // 8
                    r = i % 8
                    if indicator_in_cva:
                        for j in range(num_defs_per_path):
                            if s > tmp_exp_1[i, j]:
                                # no common-shock this time
                                def_prev = nb.int8(0)
                                for _q in range(num_cpty_buckets):
                                    def_prev |= tmp_def_indicators[_q, j] & ((1 & (_q == q)) << r)
                                if not nb.bool_(def_prev):
                                    for _q in range(num_cpty_buckets):
                                        tmp_def_indicators[_q, j] |= ((1 & (_q == q)) << r)
                                    mtm = tmp_mtm_by_cpty[i]
                                    if mtm > 0:
                                        tmp_cva_payoff_by_cpty[i, j] = discount_factor * mtm
                    else:
                        s_prev = tmp_spread_integrals_prev[i+1]
                        mtm = tmp_mtm_by_cpty[i]
                        cva_payoff_increment = discount_factor * mtm * (math.exp(-s_prev) - math.exp(-s))
                        for j in range(num_defs_per_path):
                            def_at_start = nb.int8(0)
                            for _q in range(num_cpty_buckets):
                                def_at_start |= def_indicators[coarse_start_idx, _q, j, block] & ((1 & (_q == q)) << r)
                            if (not def_at_start) and (cva_payoff_increment > 0):
                                tmp_cva_payoff_by_cpty[i, j] += cva_payoff_increment
                
            
        # Reduction routine for nested Monte-Carlo CVA
        # (assumes that inner simulations are entirely handled by one block)
        if tidx == 0:
            for j in range(num_defs_per_path):
                out1[j, block] = 0
                out2[j, block] = 0
        for i in range(num_cpty):
            for j in range(num_defs_per_path):
                if tidx < num_inner_paths:
                    c = tmp_cva_payoff_by_cpty[i, j]
                    cva_payoff_by_cpty_sh[tidx] = c
                    cva_payoff_by_cpty_sq_sh[tidx] = c
                    for k in range(i):
                        cva_payoff_by_cpty_sq_sh[tidx] += 2 * tmp_cva_payoff_by_cpty[k, j]
                    cva_payoff_by_cpty_sq_sh[tidx] *= c
                cuda.syncthreads()
                k = inner_stride // 2
                while k > 0:
                    if tidx < k:
                        cva_payoff_by_cpty_sh[tidx] += cva_payoff_by_cpty_sh[tidx + k]
                        cva_payoff_by_cpty_sq_sh[tidx] += cva_payoff_by_cpty_sq_sh[tidx + k]
                    cuda.syncthreads()
                    k //= 2
                if tidx == 0:
                    out1[j, block] += cva_payoff_by_cpty_sh[0] / num_inner_paths
                    out2[j, block] += cva_payoff_by_cpty_sq_sh[0] / num_inner_paths
                cuda.syncthreads()
                    
    _cuda_nested_cva._func.get().cache_config(prefer_shared=True)
    cuda_nested_cva = _cuda_nested_cva[num_paths, inner_stride, stream]
    
    # finally, return the compiled kernel
    return cuda_nested_cva

def compile_cuda_nested_im(irs_batch_size, vanilla_batch_size, g_diff_params, g_R, g_L_T, num_fine_per_coarse, num_rates, num_spreads, num_defs_per_path, num_paths, num_inner_paths, max_coarse_per_reset, stream):
    # compile-time constants
    num_cpty = num_spreads - 1
    num_diffusions = 2*num_rates+num_spreads-1
    fx_start = num_rates
    fx_params_start = 3*num_rates
    drift_adj_start = 4*num_rates - 1
    spread_start = fx_start + num_rates - 1
    spread_params_start = fx_params_start + 2*num_rates - 2

    if num_inner_paths & (num_inner_paths-1) == 0:
        inner_stride = num_inner_paths
    else:
        c = 0
        d = num_inner_paths
        while d != 0:
            d >>= 1
            c += 1
        inner_stride = 1 << c

    sig = (nb.float32, nb.bool_, nb.float32, nb.int32, nb.int32, nb.float32, nb.float32[:, :, :], nb.float32[:, :], nb.float32[:, :], nb.int32[:, :], nb.float32[:, :], nb.int32[:, :], nb.bool_[:, :], nb.from_dtype(xoroshiro128p_dtype)[:], nb.float32, nb.float32[:, :], nb.float32[:, :], nb.float32[:, :], nb.float32[:, :], nb.float32, nb.float32, nb.int32)

    @cuda.jit(func_or_sig=sig, max_registers=64)
    def _cuda_nested_im(alpha, adam_init, step_size, coarse_start_idx, num_coarse_steps, t, X, mtm_by_cpty, irs_f32, irs_i32, vanillas_on_fx_f32, vanillas_on_fx_i32, vanillas_on_fx_b8, rng_states, dt, out1, out2, out3, out4, adam_b1, adam_b2, adam_iter):
        block = cuda.blockIdx.x
        block_size = cuda.blockDim.x
        tidx = cuda.threadIdx.x
        pos = tidx + block * block_size

        diff_params = cuda.const.array_like(g_diff_params)
        R = cuda.const.array_like(g_R)
        irs_f32_sh = cuda.shared.array(shape=(irs_batch_size, 4), dtype=nb.float32)
        irs_i32_sh = cuda.shared.array(shape=(irs_batch_size, 3), dtype=nb.int32)
        vanillas_on_fx_f32_sh = cuda.shared.array(shape=(vanilla_batch_size, 3), dtype=nb.float32)
        vanillas_on_fx_i32_sh = cuda.shared.array(shape=(vanilla_batch_size, 2), dtype=nb.int32)
        vanillas_on_fx_b8_sh = cuda.shared.array(shape=(vanilla_batch_size, 1), dtype=nb.bool_)
        L_T = cuda.const.array_like(g_L_T)
        dW_corr = cuda.local.array(spread_start, nb.float32)
        tmp_X = cuda.local.array(spread_start, nb.float32)
        tmp_rates_sliding_window = cuda.local.array((max_coarse_per_reset, num_rates), nb.float32)
        tmp_mtm_increment_by_cpty = cuda.local.array(num_cpty, nb.float32)
        grad_sh = cuda.shared.array(inner_stride, nb.float32)

        if tidx < num_inner_paths:
            sqrt_dt = math.sqrt(dt)

            for i in range(spread_start):
                tmp_X[i] = X[coarse_start_idx+max_coarse_per_reset-1, i, block]
            
            for j in range(max_coarse_per_reset):
                for i in range(num_rates):
                    tmp_rates_sliding_window[max_coarse_per_reset-j-1, i] = X[coarse_start_idx+max_coarse_per_reset-2-j, i, block]
            
            tmp_dom_rate_integral = nb.float32(0)

            for cpty in range(num_cpty):
                tmp_mtm_increment_by_cpty[cpty] = - mtm_by_cpty[cpty, block]

            for coarse_idx in range(coarse_start_idx, coarse_start_idx+num_coarse_steps):
                if pos==0:
                    print('[ INNER | rate 0 | t =', t, '] r =', tmp_X[0], '| tmp_rates_sliding_window = (', tmp_rates_sliding_window[0, 0], '|', tmp_rates_sliding_window[1, 0], ')')
                discount_factor = math.exp(-tmp_dom_rate_integral)

                # TODO: do it also for calls just in case calls expire inside the IM window
                for batch_idx in range((irs_f32.shape[0]+irs_batch_size-1)//irs_batch_size):
                    cuda.syncthreads()
                    if tidx == 0:
                        for i in range(irs_batch_size):
                            if batch_idx*irs_batch_size+i < irs_f32.shape[0]:
                                for j in range(irs_f32.shape[1]):
                                    irs_f32_sh[i, j] = irs_f32[batch_idx*irs_batch_size+i, j]
                                for j in range(irs_i32.shape[1]):
                                    irs_i32_sh[i, j] = irs_i32[batch_idx*irs_batch_size+i, j]
                            else:
                                i -= 1
                                break
                    else:
                        i = min(irs_f32.shape[0]-batch_idx*irs_batch_size, irs_batch_size)-1
                    cuda.syncthreads()
                    for j in range(i+1):
                        first_reset = irs_f32_sh[j, 0]
                        reset_freq = irs_f32_sh[j, 1]
                        num_resets = irs_i32_sh[j, 0]
                        if first_reset + (num_resets - 1) * reset_freq + 0.1 * dt < t:
                            continue
                        notional = irs_f32_sh[j, 2]
                        cpty = irs_i32_sh[j, 1]
                        ccy = irs_i32_sh[j, 2]
                        fx = nb.float32(1)
                        if ccy != 0:
                            fx = tmp_X[num_rates + ccy - 1]
                        a = diff_params[ccy]
                        b = diff_params[num_rates+ccy]
                        sigma = diff_params[2*num_rates+ccy]
                        swap_rate = irs_f32_sh[j, 3]
                        if t > first_reset - 0.1*dt:
                            m = int((t - first_reset - (num_fine_per_coarse-1)*dt) / reset_freq) # locate the strictly previous reset date in the resets grid
                            m = int((t-first_reset-m*reset_freq+dt)/(num_fine_per_coarse*dt)) # locate it now in the coarse grid
                            m = nb.int32(max_coarse_per_reset-m)
                        else:
                            m = nb.int32(max_coarse_per_reset-1)
                        if batch_idx==0 and j==0 and pos==0:
                            print('[ t =', t, '| m =', m, ']')
                        k = int((t-first_reset+0.1*dt)/reset_freq)
                        is_coupon_date = (k >= 1) and (abs(t-first_reset-k*reset_freq) < 0.1*dt)
                        if is_coupon_date:
                            for _cpty in range(num_cpty):
                                cashflow = _cuda_price_zc_bond_inv(ccy, tmp_rates_sliding_window[m, ccy], 0, reset_freq, a, b, sigma) - 1 - swap_rate * reset_freq
                                tmp_mtm_increment_by_cpty[_cpty] += notional * fx * cashflow * (_cpty == cpty) * discount_factor
                
                for i in range(num_rates-1):
                    tmp_X[fx_start+i] = math.log(tmp_X[fx_start+i])

                for i in range(num_rates):
                    for j in range(max_coarse_per_reset-1):
                        tmp_rates_sliding_window[j, i] = tmp_rates_sliding_window[j+1, i]
                    tmp_rates_sliding_window[max_coarse_per_reset-1, i] = tmp_X[i]
                
                for fine_idx in range(num_fine_per_coarse):
                    for i in range(spread_start):
                        dW_corr[i] = 0

                    for i in range(spread_start):
                        u = xoroshiro128p_uniform_float32(rng_states, num_paths*num_defs_per_path+pos)
                        v = xoroshiro128p_uniform_float32(rng_states, num_paths*num_defs_per_path+pos)
                        v = math.sqrt(-2*math.log(u)) * math.cos(2*math.pi*v) * sqrt_dt # Box-Muller, throwing the other normal away
                        for j in range(i, spread_start):
                            # L_T is the transpose of the lower-triangular L such that Corr=L*L_T
                            dW_corr[j] += L_T[i*num_diffusions-i*(i+1)//2+j] * v
                    # E[Au*(Au)^T] = E[A*u*u^T*A^T] = A*Cov*A^T -> for unit Cov, it is enough to choose A=L
                    # E[LdW*(LdW)^T] = dt*LL^T = dt*Corr
                    # dW_corr[k] = sum_j L_{i,j} * dW_j

                    # FX log-diffusions
                    for i in range(num_rates-1):
                        tmp_X[fx_start+i] += (tmp_X[0] - tmp_X[i+1] - 0.5*diff_params[fx_params_start+i]** 2) * dt + diff_params[fx_params_start+i] * dW_corr[fx_start+i]

                    # rate diffusions
                    # TODO: change this and diffuse jointly the short rate and its integral exactly
                    # (but for now let's just stick with a numerical integral)
                    tmp_dom_rate_integral += 0.5 * tmp_X[0] * dt

                    for i in range(num_rates):
                        tmp_X[i] += diff_params[i] * \
                            (diff_params[num_rates+i] - tmp_X[i]) * dt
                        drift_adj = nb.float32(0)
                        if i != 0:
                            drift_adj = diff_params[drift_adj_start+i-1]
                        tmp_X[i] += diff_params[2*num_rates+i] * (dW_corr[i] + drift_adj * dt)

                    tmp_dom_rate_integral += 0.5 * tmp_X[0] * dt

                for i in range(num_rates-1):
                    tmp_X[fx_start+i] = math.exp(tmp_X[fx_start+i])
                
                t += dt * num_fine_per_coarse
            
            if pos==0:
                print('--')

            discount_factor = math.exp(-tmp_dom_rate_integral)
            for batch_idx in range((vanillas_on_fx_f32.shape[0]+vanilla_batch_size-1)//vanilla_batch_size):
                cuda.syncthreads()
                if tidx == 0:
                    for i in range(vanilla_batch_size):
                        if batch_idx*vanilla_batch_size+i < vanillas_on_fx_f32.shape[0]:
                            for j in range(vanillas_on_fx_f32.shape[1]):
                                vanillas_on_fx_f32_sh[i, j] = vanillas_on_fx_f32[batch_idx*vanilla_batch_size+i, j]
                            for j in range(vanillas_on_fx_i32.shape[1]):
                                vanillas_on_fx_i32_sh[i, j] = vanillas_on_fx_i32[batch_idx*vanilla_batch_size+i, j]
                            for j in range(vanillas_on_fx_b8.shape[1]):
                                vanillas_on_fx_b8_sh[i, j] = vanillas_on_fx_b8[batch_idx*vanilla_batch_size+i, j]
                        else:
                            i -= 1
                            break
                else:
                    i = min(vanillas_on_fx_f32.shape[0]-batch_idx*vanilla_batch_size, vanilla_batch_size)-1
                cuda.syncthreads()
                for j in range(i+1):
                    maturity = vanillas_on_fx_f32_sh[j, 0]
                    if maturity + 0.1 * dt < t:
                        continue
                    notional = vanillas_on_fx_f32_sh[j, 1]
                    strike = vanillas_on_fx_f32_sh[j, 2]
                    cpty = vanillas_on_fx_i32_sh[j, 0]
                    undl = vanillas_on_fx_i32_sh[j, 1]
                    call_put = vanillas_on_fx_b8_sh[j, 0]
                    a_d = diff_params[0]
                    a_f = diff_params[undl]
                    b_d = diff_params[num_rates]
                    b_f = diff_params[num_rates+undl]
                    s_d = diff_params[2*num_rates]
                    s_f = diff_params[2*num_rates+undl]
                    s_fx = diff_params[3*num_rates+undl-1]
                    price = _cuda_price_vanilla_on_fx(call_put, strike, t, maturity, tmp_X[0], tmp_X[undl],
                                                    tmp_X[num_rates+undl-1], R[num_rates+undl-1],
                                                    R[undl*num_diffusions-undl*(undl+1)//2+num_rates+undl-1],
                                                    R[undl], a_d, a_f, b_d, b_f, s_d, s_f, s_fx, dt)
                    for _cpty in range(num_cpty):
                        tmp_mtm_increment_by_cpty[_cpty] += notional * price * (_cpty == cpty) * discount_factor
            
            for batch_idx in range((irs_f32.shape[0]+irs_batch_size-1)//irs_batch_size):
                cuda.syncthreads()
                if tidx == 0:
                    for i in range(irs_batch_size):
                        if batch_idx*irs_batch_size+i < irs_f32.shape[0]:
                            for j in range(irs_f32.shape[1]):
                                irs_f32_sh[i, j] = irs_f32[batch_idx*irs_batch_size+i, j]
                            for j in range(irs_i32.shape[1]):
                                irs_i32_sh[i, j] = irs_i32[batch_idx*irs_batch_size+i, j]
                        else:
                            i -= 1
                            break
                else:
                    i = min(irs_f32.shape[0]-batch_idx*irs_batch_size, irs_batch_size)-1
                cuda.syncthreads()
                for j in range(i+1):
                    first_reset = irs_f32_sh[j, 0]
                    reset_freq = irs_f32_sh[j, 1]
                    num_resets = irs_i32_sh[j, 0]
                    if first_reset + (num_resets - 1) * reset_freq + 0.1 * dt < t:
                        continue
                    notional = irs_f32_sh[j, 2]
                    cpty = irs_i32_sh[j, 1]
                    ccy = irs_i32_sh[j, 2]
                    fx = nb.float32(1)
                    if ccy != 0:
                        fx = tmp_X[num_rates + ccy - 1]
                    a = diff_params[ccy]
                    b = diff_params[num_rates+ccy]
                    sigma = diff_params[2*num_rates+ccy]
                    swap_rate = irs_f32_sh[j, 3]
                    if t > first_reset - 0.1*dt:
                        m = int((t - first_reset - (num_fine_per_coarse-1)*dt) / reset_freq) # locate the strictly previous reset date in the resets grid
                        m = int((t-first_reset-m*reset_freq+dt)/(num_fine_per_coarse*dt)) # locate it now in the coarse grid
                        m = nb.int32(max_coarse_per_reset-m)
                    else:
                        m = nb.int32(max_coarse_per_reset-1)
                    price = _cuda_price_irs(ccy, swap_rate, tmp_rates_sliding_window[m, ccy], tmp_X[ccy], t, first_reset, reset_freq, num_resets, False, a, b, sigma, dt)
                    for _cpty in range(num_cpty):
                        tmp_mtm_increment_by_cpty[_cpty] += notional * fx * price * (_cpty == cpty) * discount_factor
        
        # scalar SGD iteration for the nested quantile
        for c in range(num_cpty):
            if adam_init:
                # 1st moment
                if tidx < num_inner_paths:
                    grad_sh[tidx] = tmp_mtm_increment_by_cpty[c]
                cuda.syncthreads()
                k = inner_stride // 2
                while k > 0:
                    if tidx < k:
                        grad_sh[tidx] += grad_sh[tidx + k]
                    cuda.syncthreads()
                    k //= 2
                if tidx == 0:
                    grad_sh[0] /= num_inner_paths
                    out1[c, block] = grad_sh[0]
                cuda.syncthreads()
                tmp_quantile = grad_sh[0]

                # 2nd moment
                if tidx < num_inner_paths:
                    grad_sh[tidx] = tmp_mtm_increment_by_cpty[c]**2
                cuda.syncthreads()
                k = inner_stride // 2
                while k > 0:
                    if tidx < k:
                        grad_sh[tidx] += grad_sh[tidx + k]
                    cuda.syncthreads()
                    k //= 2
                if tidx == 0:
                    grad_sh[0] = math.sqrt(grad_sh[0] / num_inner_paths - tmp_quantile**2)
                    out2[c, block] = grad_sh[0]
                cuda.syncthreads()
                tmp_std = grad_sh[0]
            else:
                tmp_quantile = out1[c, block]
            
            if tidx < num_inner_paths:
                grad_sh[tidx] = alpha
                if tmp_mtm_increment_by_cpty[c] > tmp_quantile:
                    grad_sh[tidx] -= 1
            cuda.syncthreads()
            k = inner_stride // 2
            while k > 0:
                if tidx < k:
                    grad_sh[tidx] += grad_sh[tidx + k]
                cuda.syncthreads()
                k //= 2
            if pos == 0:
                tmp_std = out2[c, block]
                print('path =', block, '| t =', t-num_fine_per_coarse*num_coarse_steps*dt, '| avg grad =', grad_sh[0] / num_inner_paths, '| quantile =', tmp_quantile, '| tmp_std =', tmp_std)
            if tidx == 0:
                if not adam_init:
                    tmp_std = out2[c, block]
                    tmp_m = adam_b1 * out3[c, block] + (1 - adam_b1) * grad_sh[0] / num_inner_paths
                    tmp_v = adam_b2 * out4[c, block] + (1 - adam_b2) * (grad_sh[0] / num_inner_paths)**2
                else:
                    tmp_m = (1 - adam_b1) * grad_sh[0] / num_inner_paths
                    tmp_v = (1 - adam_b2) * (grad_sh[0] / num_inner_paths)**2
                out3[c, block] = tmp_m
                out4[c, block] = tmp_v
                tmp_m /= 1 - adam_b1**(adam_iter+1)
                tmp_v /= 1 - adam_b2**(adam_iter+1)
                out1[c, block] -= step_size * tmp_std * tmp_m / (math.sqrt(tmp_v)+1e-8)
            cuda.syncthreads()
    
    _cuda_nested_im._func.get().cache_config(prefer_shared=True)
    cuda_nested_im = _cuda_nested_im[num_paths, inner_stride, stream]
    
    # finally, return the compiled kernel
    return cuda_nested_im

@cuda.jit(device=True, inline=True)
def _cuda_norm_cdf(z):
    return 0.5*(1.+math.erf(z/math.sqrt(2.)))


@cuda.jit(device=True, inline=True)
def _cuda_price_zc_bond(ccy, r_t, t, mat, a, b, sigma):
    B = (1.-math.exp(-a*(mat-t)))/a
    A = (b-0.5*sigma*sigma/(a*a))*(B-mat+t)-0.25*sigma*sigma/a*B*B
    return math.exp(A-B*r_t)

@cuda.jit(device=True, inline=True)
def _cuda_price_zc_bond_inv(ccy, r_t, t, mat, a, b, sigma):
    B = (1.-math.exp(-a*(mat-t)))/a
    A = (b-0.5*sigma*sigma/(a*a))*(B-mat+t)-0.25*sigma*sigma/a*B*B
    return math.exp(B*r_t-A)

@cuda.jit(device=True, inline=True)
def _cuda_price_irs(ccy, swap_rate, r_prev_reset, r_t, t, first_reset, reset_freq, num_resets, only_fixed_leg, a, b, sigma, dt):
    if(t > first_reset+(num_resets-1)*reset_freq+0.1*dt):
        return nb.float32(0)
    fixed_leg = nb.float32(0)
    k = int((t-first_reset+0.1*dt)/reset_freq)
    if k < 0:
        k = nb.int32(0)
    reset = first_reset+k*reset_freq
    zc_last = nb.float32(1.)
    for i in range(k+1, num_resets):
        reset += reset_freq
        zc_last = _cuda_price_zc_bond(ccy, r_t, t, reset, a, b, sigma)
        fixed_leg += zc_last
    fixed_leg *= reset_freq * swap_rate
    if t < first_reset - 0.1*dt:
        floating_leg = _cuda_price_zc_bond(
            ccy, r_t, t, first_reset, a, b, sigma) - zc_last
    elif abs(t-first_reset-k*reset_freq) < 0.1*dt:
        if k == 0:
            floating_leg = 1 - zc_last
        else:
            floating_leg = _cuda_price_zc_bond_inv(ccy, r_prev_reset, 0, reset_freq, a, b, sigma) - zc_last
            fixed_leg += reset_freq * swap_rate
    else:
        t_next_reset = first_reset + (k+1) * reset_freq
        floating_leg = _cuda_price_zc_bond(ccy, r_t, t, t_next_reset, a, b, sigma)*_cuda_price_zc_bond_inv(ccy, r_prev_reset,
                                                                                                       0,
                                                                                                       reset_freq, a, b,
                                                                                                       sigma) - zc_last
    if only_fixed_leg:
        return fixed_leg
    else:
        return floating_leg - fixed_leg

@cuda.jit(device=True, inline=True)
def _cuda_price_vanilla_on_fx(call_put, stk, t, mat, r_d_t, r_f_t, fx_t, rho_fx_d, rho_fx_f, rho_f_d, a_d, a_f, b_d, b_f,
                              s_d, s_f, s_fx, dt):
    if abs(t-mat) < 0.1*dt:
        return max(fx_t - stk, 0.)
    zc_d = _cuda_price_zc_bond(0, r_d_t, t, mat, a_d, b_d, s_d)
    zc_f = _cuda_price_zc_bond(0, r_f_t, t, mat, a_f, b_f, s_f)
    e_d_resmat = math.exp(-a_d*(mat-t))
    e_f_resmat = math.exp(-a_f*(mat-t))
    int_B_d_sq = (mat-t+(2*e_d_resmat-0.5*e_d_resmat *
                         e_d_resmat-1.5)/a_d)/(a_d*a_d)
    int_B_f_sq = (mat-t+(2*e_f_resmat-0.5*e_f_resmat *
                         e_f_resmat-1.5)/a_f)/(a_f*a_f)
    int_B_d = (mat-t+(e_d_resmat-1)/a_d)/a_d
    int_B_f = (mat-t+(e_f_resmat-1)/a_f)/a_f
    int_B_d_B_f = ((mat-t)+(e_f_resmat+e_d_resmat-e_d_resmat*e_f_resmat-1)/(a_d+a_f)) / \
        a_d+(e_f_resmat-1)/(a_f*(a_d+a_f))+(e_d_resmat-1)/(a_d*a_d*(a_d+a_f))
    pricing_vol = math.sqrt(s_fx*s_fx*(mat-t)+s_d*s_d*int_B_d_sq+s_f*s_f*int_B_f_sq+2 *
                            rho_fx_d*s_fx*s_d*int_B_d-2*rho_f_d*s_f*s_d*int_B_d_B_f-2*rho_fx_f*s_fx*s_f*int_B_f)
    d_1 = math.log(fx_t/stk*zc_f/zc_d)/pricing_vol+0.5*pricing_vol
    d_2 = d_1-pricing_vol
    if call_put:
        return zc_f*fx_t*_cuda_norm_cdf(d_1)-zc_d*stk*_cuda_norm_cdf(d_2)
    else:
        return -zc_f*fx_t*_cuda_norm_cdf(-d_1)+zc_d*stk*_cuda_norm_cdf(-d_2)

def compile_cuda_compute_mtm(irs_batch_size, vanilla_batch_size, g_diff_params, g_R, num_fine_per_coarse, num_rates, num_spreads,
                             num_paths, ntpb, stream):
    # compile-time constants
    num_diffusions = 2*num_rates+num_spreads-1

    sig = (nb.int32, nb.float32, nb.float32[:, :, :], nb.float32[:, :, :], nb.float32[:, :, :], 
           nb.float32[:, :], nb.int32[:, :], nb.bool_[:, :], 
           nb.float32[:, :], nb.int32[:, :], nb.float32[:, :], nb.int32[:, :], 
           nb.float32, nb.int32, nb.int32, nb.bool_)
    @cuda.jit(func_or_sig=sig, max_registers=64)
    def _cuda_compute_mtm(coarse_idx, t, X, mtm_by_cpty, cash_flows_by_cpty, vanillas_on_fx_f32, vanillas_on_fx_i32, vanillas_on_fx_b8, irs_f32, irs_i32, zcs_f32, zcs_i32, dt, max_coarse_per_reset, window_length, set_irs_at_par):
        block = cuda.blockIdx.x
        block_size = cuda.blockDim.x
        tidx = cuda.threadIdx.x
        pos = tidx + block * block_size

        if pos < num_paths:
            # TODO: price on a range of coarse steps, should minimize global->shared copies
            diff_params = cuda.const.array_like(g_diff_params)
            R = cuda.const.array_like(g_R)
            irs_f32_sh = cuda.shared.array(shape=(irs_batch_size, 4), dtype=nb.float32)
            irs_i32_sh = cuda.shared.array(shape=(irs_batch_size, 3), dtype=nb.int32)
            vanillas_on_fx_f32_sh = cuda.shared.array(shape=(vanilla_batch_size, 3), dtype=nb.float32)
            vanillas_on_fx_i32_sh = cuda.shared.array(shape=(vanilla_batch_size, 2), dtype=nb.int32)
            vanillas_on_fx_b8_sh = cuda.shared.array(shape=(vanilla_batch_size, 1), dtype=nb.bool_)

            for cpty in range(num_spreads-1):
                mtm_by_cpty[coarse_idx, cpty, pos] = 0.
                cash_flows_by_cpty[coarse_idx, cpty, pos] = 0.

            for batch_idx in range((vanillas_on_fx_f32.shape[0]+vanilla_batch_size-1)//vanilla_batch_size):
                cuda.syncthreads()
                if tidx == 0:
                    for i in range(vanilla_batch_size):
                        if batch_idx*vanilla_batch_size+i < vanillas_on_fx_f32.shape[0]:
                            for j in range(vanillas_on_fx_f32.shape[1]):
                                vanillas_on_fx_f32_sh[i, j] = vanillas_on_fx_f32[batch_idx*vanilla_batch_size+i, j]
                            for j in range(vanillas_on_fx_i32.shape[1]):
                                vanillas_on_fx_i32_sh[i, j] = vanillas_on_fx_i32[batch_idx*vanilla_batch_size+i, j]
                            for j in range(vanillas_on_fx_b8.shape[1]):
                                vanillas_on_fx_b8_sh[i, j] = vanillas_on_fx_b8[batch_idx*vanilla_batch_size+i, j]
                        else:
                            i -= 1
                            break
                else:
                    i = min(vanillas_on_fx_f32.shape[0]-batch_idx*vanilla_batch_size, vanilla_batch_size)-1
                cuda.syncthreads()
                for j in range(i+1):
                    maturity = vanillas_on_fx_f32_sh[j, 0]
                    if maturity + 0.1 * dt < t:
                        continue
                    notional = vanillas_on_fx_f32_sh[j, 1]
                    strike = vanillas_on_fx_f32_sh[j, 2]
                    cpty = vanillas_on_fx_i32_sh[j, 0]
                    undl = vanillas_on_fx_i32_sh[j, 1]
                    call_put = vanillas_on_fx_b8_sh[j, 0]
                    a_d = diff_params[0]
                    a_f = diff_params[undl]
                    b_d = diff_params[num_rates]
                    b_f = diff_params[num_rates+undl]
                    s_d = diff_params[2*num_rates]
                    s_f = diff_params[2*num_rates+undl]
                    s_fx = diff_params[3*num_rates+undl-1]
                    price = _cuda_price_vanilla_on_fx(call_put, strike, t, maturity, X[coarse_idx+max_coarse_per_reset-1, 0, pos], X[coarse_idx+max_coarse_per_reset-1, undl, pos],
                                                    X[coarse_idx+max_coarse_per_reset-1, num_rates+undl-1, pos], R[num_rates+undl-1],
                                                    R[undl*num_diffusions-undl*(undl+1)//2+num_rates+undl-1],
                                                    R[undl], a_d, a_f, b_d, b_f, s_d, s_f, s_fx, dt)
                    mtm_by_cpty[coarse_idx, cpty, pos] += notional * price
                    if t > maturity - 0.1*dt:
                        if call_put:
                            payoff = X[coarse_idx+max_coarse_per_reset-1, num_rates+undl-1, pos] - strike
                        else:
                            payoff = strike - X[coarse_idx+max_coarse_per_reset-1, num_rates+undl-1, pos]
                        if payoff < 0:
                            payoff = 0
                        cash_flows_by_cpty[coarse_idx, cpty, pos] += notional * payoff

            # FIXME: to be tested, this is a first implementation of using shared mem as fast buffer
            # for IRS specs
            for batch_idx in range((irs_f32.shape[0]+irs_batch_size-1)//irs_batch_size):
                cuda.syncthreads()
                if tidx == 0:
                    for i in range(irs_batch_size):
                        if batch_idx*irs_batch_size+i < irs_f32.shape[0]:
                            for j in range(irs_f32.shape[1]):
                                irs_f32_sh[i, j] = irs_f32[batch_idx*irs_batch_size+i, j]
                            for j in range(irs_i32.shape[1]):
                                irs_i32_sh[i, j] = irs_i32[batch_idx*irs_batch_size+i, j]
                        else:
                            i -= 1
                            break
                else:
                    i = min(irs_f32.shape[0]-batch_idx*irs_batch_size, irs_batch_size)-1
                cuda.syncthreads()
                for j in range(i+1):
                    first_reset = irs_f32_sh[j, 0]
                    reset_freq = irs_f32_sh[j, 1]
                    num_resets = irs_i32_sh[j, 0]
                    if first_reset + (num_resets - 1) * reset_freq + 0.1 * dt < t:
                        continue
                    notional = irs_f32_sh[j, 2]
                    cpty = irs_i32_sh[j, 1]
                    ccy = irs_i32_sh[j, 2]
                    fx = nb.float32(1)
                    if ccy != 0:
                        fx = X[coarse_idx+max_coarse_per_reset-1, num_rates + ccy - 1, pos]
                    a = diff_params[ccy]
                    b = diff_params[num_rates+ccy]
                    sigma = diff_params[2*num_rates+ccy]
                    if set_irs_at_par and coarse_idx==0:
                        fixed = _cuda_price_irs(ccy, 1., 0., X[max_coarse_per_reset-1, ccy, pos], 0., first_reset, reset_freq,
                                                num_resets, True, a, b, sigma, dt)
                        floating = _cuda_price_irs(ccy, 0., 0., X[max_coarse_per_reset-1, ccy, pos], 0., first_reset, reset_freq,
                                                num_resets, False, a, b, sigma, dt)
                        swap_rate = floating/fixed
                        if pos == 0:
                            irs_f32[batch_idx*irs_batch_size+j, 3] = swap_rate
                    else:
                        swap_rate = irs_f32_sh[j, 3]
                    if t > first_reset - 0.1*dt:
                        num_coarse_per_reset = int((reset_freq+dt)/(num_fine_per_coarse*dt))
                        m = int((t - first_reset - (num_fine_per_coarse-1)*dt) / reset_freq) # locate the strictly previous reset date in the resets grid
                        m = int((first_reset+m*reset_freq+dt)/(num_fine_per_coarse*dt)) # locate it now in the coarse grid
                        # TODO: given window_length and last_reset_idx, deduce local_last_reset_idx
                        m = (m-coarse_idx+num_coarse_per_reset) % window_length + coarse_idx-num_coarse_per_reset # locate in (extended) local window
                    else:
                        m = 0
                    price = _cuda_price_irs(ccy, swap_rate, X[m+max_coarse_per_reset-1, ccy, pos], X[coarse_idx+max_coarse_per_reset-1, ccy, pos], t, first_reset, reset_freq, num_resets, False, a, b, sigma, dt)
                    mtm_by_cpty[coarse_idx, cpty, pos] += notional * fx * price
                    k = int((t-first_reset+0.1*dt)/reset_freq)
                    is_coupon_date = (k >= 1) and (abs(t-first_reset-k*reset_freq) < 0.1*dt)
                    if is_coupon_date:
                        cash_flows_by_cpty[coarse_idx, cpty, pos] += notional * fx * (_cuda_price_zc_bond_inv(ccy, X[m+max_coarse_per_reset-1, ccy, pos], 0, reset_freq, a, b, sigma) - 1 - swap_rate * reset_freq)

    _cuda_compute_mtm._func.get().cache_config(prefer_shared=True)
    cuda_compute_mtm = _cuda_compute_mtm[(num_paths+ntpb-1)//ntpb, ntpb, stream]

    # returning the compiled kernel
    return cuda_compute_mtm
