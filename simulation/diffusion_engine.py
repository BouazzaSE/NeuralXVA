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

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
import numexpr as ne
from simulation.kernels import compile_cuda_bulk_diffuse, compile_cuda_compute_mtm, compile_cuda_diffuse_and_price, compile_cuda_oversimulate_defs, compile_cuda_generate_exp1, compile_cuda_nested_cva, compile_cuda_nested_im


class DiffusionEngine:
    def __init__(self, irs_batch_size, vanilla_batch_size, num_coarse_steps, dT, num_fine_per_coarse, dt, num_paths, num_inner_paths, 
                 num_defs_per_path, num_rates, num_spreads, R, rates_params, fx_params,
                 spreads_params, vanilla_specs, irs_specs, zcs_specs,
                 initial_values, initial_defaults, cDtoH_freq, device=0, no_nested_cva=False, no_nested_im=False, num_adam_iters=100, lam=1, gamma=0.5, adam_b1=0.9, adam_b2=0.999):
        cuda.select_device(device)
        self.irs_batch_size = irs_batch_size
        self.vanilla_batch_size = vanilla_batch_size
        self.num_coarse_steps = num_coarse_steps
        self.num_fine_per_coarse = num_fine_per_coarse
        self.num_paths = num_paths
        self.num_inner_paths = num_inner_paths
        self.num_defs_per_path = num_defs_per_path
        self.num_rates = num_rates
        self.num_spreads = num_spreads
        self.vanilla_specs = vanilla_specs.copy()
        self.irs_specs = irs_specs.copy()
        self.zcs_specs = zcs_specs.copy()
        self.cDtoH_freq = cDtoH_freq
        self.no_nested_cva = no_nested_cva
        self.no_nested_im = no_nested_im
        self.num_adam_iters = num_adam_iters
        self.lam = lam
        self.gamma = gamma
        self.adam_b1 = adam_b1
        self.adam_b2 = adam_b2

        if irs_specs.size > 0:
            self.max_coarse_per_reset = max(int((self.irs_specs['reset_freq'].max()+dt)/dT), 1)
        else:
            self.max_coarse_per_reset = 1
        # TODO: add assert statements on the acceptable range for reset_freq

        # force casting of float constants
        self.dt = np.float32(dt)
        self.dT = np.float32(dT)

        self.num_diffusions = 2*self.num_rates+self.num_spreads-1

        self.stream = cuda.stream()

        self._allocate_host_arrays()
        self._set_cpu_arrays(R, rates_params, fx_params, spreads_params,
                             initial_values, initial_defaults)
        self._allocate_device_arrays()
        self._copy_product_specs_to_device()

        self.cuda_generate_exp1 = compile_cuda_generate_exp1(self.num_spreads,
                                                             self.num_defs_per_path,
                                                             self.num_paths,
                                                             512,
                                                             self.stream)
        self.cuda_bulk_diffuse = compile_cuda_bulk_diffuse(self.g_diff_params,
                                                           self.g_L_T,
                                                           self.num_fine_per_coarse,
                                                           self.num_rates,
                                                           self.num_spreads,
                                                           self.num_defs_per_path,
                                                           self.num_paths,
                                                           512, self.stream)
        self.cuda_compute_mtm = compile_cuda_compute_mtm(self.irs_batch_size, 
                                                         self.vanilla_batch_size,
                                                         self.g_diff_params,
                                                         self.g_R, 
                                                         self.num_fine_per_coarse,
                                                         self.num_rates,
                                                         self.num_spreads,
                                                         self.num_paths, 512,
                                                         self.stream)
        self.cuda_diffuse_and_price = compile_cuda_diffuse_and_price(self.irs_batch_size, 
                                                         self.vanilla_batch_size,
                                                         self.g_diff_params,
                                                         self.g_R,
                                                         self.g_L_T,
                                                         self.num_fine_per_coarse,
                                                         self.num_rates,
                                                         self.num_spreads,
                                                         self.num_paths, 
                                                         512,
                                                         self.stream)
        self.cuda_oversimulate_defs = compile_cuda_oversimulate_defs(self.num_spreads,
                                                         self.num_defs_per_path,
                                                         self.num_paths, 
                                                         512,
                                                         self.stream)
        self.cuda_nested_cva = compile_cuda_nested_cva(self.irs_batch_size, 
                                                       self.vanilla_batch_size,
                                                       self.g_diff_params, 
                                                       self.g_R, 
                                                       self.g_L_T, 
                                                       self.num_fine_per_coarse, 
                                                       self.num_rates, 
                                                       self.num_spreads, 
                                                       self.num_defs_per_path,
                                                       self.num_paths, 
                                                       self.num_inner_paths, 
                                                       self.max_coarse_per_reset,
                                                       self.stream)
        self.cuda_nested_im = compile_cuda_nested_im(self.irs_batch_size, 
                                                       self.vanilla_batch_size,
                                                       self.g_diff_params, 
                                                       self.g_R, 
                                                       self.g_L_T, 
                                                       self.num_fine_per_coarse, 
                                                       self.num_rates, 
                                                       self.num_spreads, 
                                                       self.num_defs_per_path,
                                                       self.num_paths, 
                                                       self.num_inner_paths, 
                                                       self.max_coarse_per_reset,
                                                       self.stream)
        print('Successfully compiled all kernels.')
        self.d_rng_states = None
        self.reset_rng_states()

    def _allocate_host_arrays(self):
        self.X = cuda.pinned_array(
            (self.num_coarse_steps+1, self.num_diffusions, self.num_paths), np.float32)
        self.mtm_by_cpty = cuda.pinned_array(
            (self.num_coarse_steps+1, self.num_spreads-1, self.num_paths), np.float32)
        self.cash_flows_by_cpty = cuda.pinned_array(
            (self.num_coarse_steps+1, self.num_spreads-1, self.num_paths), np.float32)
        self.spread_integrals = cuda.pinned_array(
            (self.num_coarse_steps+1, self.num_spreads, self.num_paths), np.float32)
        self.dom_rate_integral = cuda.pinned_array(
            (self.num_coarse_steps+1, self.num_paths), np.float32)
        self.def_indicators = cuda.pinned_array(
            (self.num_coarse_steps+1, (self.num_spreads-1+7)//8, self.num_defs_per_path, self.num_paths), 
            np.int8)
        if not self.no_nested_cva:
            try:
                self.nested_cva = cuda.pinned_array(
                    (self.num_coarse_steps+1, self.num_defs_per_path, self.num_paths), np.float32)
            except cuda.cudadrv.driver.CudaAPIError:
                print('couldn\'t allocate pinned array for nested_cva, using the numpy allocator instead (non-pinned array).')
                self.nested_cva = np.empty((self.num_coarse_steps+1, self.num_defs_per_path, self.num_paths), np.float32)
            try:
                self.nested_cva_sq = cuda.pinned_array(
                    (self.num_coarse_steps+1, self.num_defs_per_path, self.num_paths), np.float32)
            except cuda.cudadrv.driver.CudaAPIError:
                print('couldn\'t allocate pinned array for nested_cva_sq, using the numpy allocator instead (non-pinned array).')
                self.nested_cva_sq = np.empty((self.num_coarse_steps+1, self.num_defs_per_path, self.num_paths), np.float32)
        
        if not self.no_nested_im:
            try:
                self.nested_im_by_cpty = cuda.pinned_array(
                    (self.num_coarse_steps+1, self.num_spreads-1, self.num_paths), np.float32)
            except cuda.cudadrv.driver.CudaAPIError:
                print('couldn\'t allocate pinned array for nested_im_by_cpty, using the numpy allocator instead (non-pinned array).')
                self.nested_im_by_cpty = np.empty((self.num_coarse_steps+1, self.num_spreads-1, self.num_paths), np.float32)

        self.R = np.empty(
            (self.num_diffusions, self.num_diffusions), dtype=np.float32)
        # following matrices are flattened (and only upper-triangular entries are kept)
        self.g_R = np.empty(self.num_diffusions * (self.num_diffusions+1)//2, np.float32)
        self.g_L_T = np.empty(self.num_diffusions * (self.num_diffusions+1)//2, np.float32)

        self.g_diff_params = np.empty(5*self.num_rates-2+3*self.num_spreads, np.float32)

        # storing copies of product specs for each warp
        self.vanillas_on_fx_f32 = np.empty((self.vanilla_specs.size, 3), np.float32)  # mat, notional, stk
        self.vanillas_on_fx_i32 = np.empty((self.vanilla_specs.size, 2), np.int32)  # cpty, fgn_ccy
        self.vanillas_on_fx_b8 = np.empty((self.vanilla_specs.size, 1), np.bool8)  # call_put
        # first_reset, reset_freq, notional, swap_rate
        self.irs_f32 = np.empty((self.irs_specs.size, 4), np.float32)
        # num_resets, cpty, ccy
        self.irs_i32 = np.empty((self.irs_specs.size, 3), np.int32)
        self.zcs_f32 = np.empty((self.zcs_specs.size, 2), np.float32)  # mat, notional
        self.zcs_i32 = np.empty((self.zcs_specs.size, 2), np.int32)  # cpty, ccy

    def _allocate_device_arrays(self):
        self.d_exp_1 = cuda.device_array(
            (self.num_spreads-1, self.num_defs_per_path, self.num_paths), np.float32)
        self.d_X = cuda.device_array(
            (self.cDtoH_freq+self.max_coarse_per_reset, self.num_diffusions, self.num_paths), np.float32)
        self.d_spread_integrals = cuda.device_array(
            (self.cDtoH_freq+1, self.num_spreads, self.num_paths), np.float32)
        self.d_dom_rate_integral = cuda.device_array(
            (self.cDtoH_freq+1, self.num_paths), np.float32)
        self.d_def_indicators = cuda.device_array(
            (self.cDtoH_freq+1, (self.num_spreads-1+7)//8, self.num_defs_per_path, self.num_paths), np.int8)
        if not self.no_nested_cva:
            self.d_nested_cva = cuda.device_array((self.num_defs_per_path, self.num_paths), np.float32)
            self.d_nested_cva_sq = cuda.device_array((self.num_defs_per_path, self.num_paths), np.float32)
        if not self.no_nested_im:
            self.d_nested_im_by_cpty = cuda.device_array((self.num_spreads-1, self.num_paths), np.float32)
            self.d_nested_im_std_by_cpty = cuda.device_array((self.num_spreads-1, self.num_paths), np.float32)
            self.d_nested_im_m = cuda.device_array((self.num_spreads-1, self.num_paths), np.float32)
            self.d_nested_im_v = cuda.device_array((self.num_spreads-1, self.num_paths), np.float32)
        self.d_vanillas_on_fx_f32 = cuda.device_array((self.vanilla_specs.size, 3), np.float32)
        self.d_vanillas_on_fx_i32 = cuda.device_array((self.vanilla_specs.size, 2), np.int32)
        self.d_vanillas_on_fx_b8 = cuda.device_array((self.vanilla_specs.size, 1), np.bool8)
        self.d_irs_f32 = cuda.device_array((self.irs_specs.size, 4), np.float32)
        self.d_irs_i32 = cuda.device_array((self.irs_specs.size, 3), np.int32)
        self.d_zcs_f32 = cuda.device_array(
            (self.zcs_specs.size, 2), np.float32)
        self.d_zcs_i32 = cuda.device_array(
            (self.zcs_specs.size, 2), np.int32)
        self.d_mtm_by_cpty = cuda.device_array(
            (self.cDtoH_freq+1, self.num_spreads-1, self.num_paths), np.float32)
        self.d_cash_flows_by_cpty = cuda.device_array(
            (self.cDtoH_freq+1, self.num_spreads-1, self.num_paths), np.float32)
    
    def _set_cpu_arrays(self, R, rates_params, fx_params, spreads_params,
                        initial_values, initial_defaults):
        assert R.shape[0] == R.shape[1] == 2*self.num_rates+self.num_spreads-1, \
            'incorrect shape for correlation matrix'
        assert R.dtype == np.float32, 'use only float32 for floating point numbers'
        self.R[:] = R
        triu_indices = np.triu_indices(self.num_diffusions)
        self.g_R[:] = self.R[triu_indices]
        self.g_L_T[:] = np.linalg.cholesky(self.R).T[triu_indices]

        self.g_diff_params[:self.num_rates] = rates_params['a']
        self.g_diff_params[self.num_rates:2*self.num_rates] = rates_params['b']
        self.g_diff_params[2*self.num_rates:3 *
                           self.num_rates] = rates_params['sigma']
        self.g_diff_params[3*self.num_rates:4 *
                           self.num_rates-1] = fx_params['vol']
        # BEGIN DRIFT ADJ
        self.g_diff_params[4*self.num_rates-1:5*self.num_rates-2] = -self.g_diff_params[3*self.num_rates:4 *
                                                                                        self.num_rates-1]*self.R[1:self.num_rates, self.num_rates:(2*self.num_rates-1)].diagonal()
        # END DRIFT ADJ
        self.g_diff_params[5*self.num_rates-2:5*self.num_rates +
                           self.num_spreads-2] = spreads_params['a']
        self.g_diff_params[5*self.num_rates+self.num_spreads-2:5 *
                           self.num_rates+2*self.num_spreads-2] = spreads_params['b']
        self.g_diff_params[5*self.num_rates+2 *
                           self.num_spreads-2:] = spreads_params['vvol']

        self.vanillas_on_fx_f32[:, 0] = self.vanilla_specs['maturity']
        self.vanillas_on_fx_f32[:, 1] = self.vanilla_specs['notional']
        self.vanillas_on_fx_f32[:, 2] = self.vanilla_specs['strike']
        self.vanillas_on_fx_i32[:, 0] = self.vanilla_specs['cpty']
        self.vanillas_on_fx_i32[:, 1] = self.vanilla_specs['undl']
        self.vanillas_on_fx_b8[:, 0] = self.vanilla_specs['call_put']

        self.irs_f32[:, 0] = self.irs_specs['first_reset']
        self.irs_f32[:, 1] = self.irs_specs['reset_freq']
        self.irs_f32[:, 2] = self.irs_specs['notional']
        self.irs_f32[:, 3] = self.irs_specs['swap_rate']
        self.irs_i32[:, 0] = self.irs_specs['num_resets']
        self.irs_i32[:, 1] = self.irs_specs['cpty']
        self.irs_i32[:, 2] = self.irs_specs['undl']

        self.zcs_f32[:, 0] = self.zcs_specs['maturity']
        self.zcs_f32[:, 1] = self.zcs_specs['notional']
        self.zcs_i32[:, 0] = self.zcs_specs['cpty']
        self.zcs_i32[:, 1] = self.zcs_specs['undl']

        self.spread_integrals[0] = 0.
        self.dom_rate_integral[0] = 0.
        self.def_indicators[0] = initial_defaults[:, np.newaxis, np.newaxis]

        self.X[0, :self.num_rates] = initial_values[:self.num_rates, np.newaxis]
        self.X[0, self.num_rates:(
            2*self.num_rates-1)] = initial_values[self.num_rates:(2*self.num_rates-1), np.newaxis]
        self.X[0, (2*self.num_rates-1):(2*self.num_rates+self.num_spreads-1)] = initial_values[(
            2*self.num_rates-1):(2*self.num_rates+self.num_spreads-1), np.newaxis]

    def _copy_product_specs_to_device(self):
        # copying product specs to GPU
        cuda.to_device(self.vanillas_on_fx_f32, to=self.d_vanillas_on_fx_f32)
        cuda.to_device(self.vanillas_on_fx_i32, to=self.d_vanillas_on_fx_i32)
        cuda.to_device(self.vanillas_on_fx_b8, to=self.d_vanillas_on_fx_b8)
        cuda.to_device(self.irs_f32, to=self.d_irs_f32)
        cuda.to_device(self.irs_i32, to=self.d_irs_i32)
        cuda.to_device(self.zcs_f32, to=self.d_zcs_f32)
        cuda.to_device(self.zcs_i32, to=self.d_zcs_i32)

    def _reset(self):
        cuda.to_device(self.X[0], to=self.d_X[self.max_coarse_per_reset-1], stream=self.stream)
        cuda.to_device(
            self.spread_integrals[0], to=self.d_spread_integrals[0], stream=self.stream)
        cuda.to_device(
            self.dom_rate_integral[0], to=self.d_dom_rate_integral[0], stream=self.stream)
        self.def_indicators[:] = self.def_indicators[0][None]
        cuda.to_device(self.def_indicators[:self.cDtoH_freq+1], to=self.d_def_indicators, stream=self.stream)

    def generate_batch(self, end=None, verbose=False, fused=False, nested_cva_at=None, nested_im_at=None, indicator_in_cva=False, alpha=None, im_window=None):
        if end is None:
            end = self.num_coarse_steps
        t = 0.
        self._reset()
        self.cuda_generate_exp1(self.d_exp_1, self.d_rng_states)
        self.stream.synchronize()
        self.cuda_compute_mtm(0, t, self.d_X, self.d_mtm_by_cpty, self.d_cash_flows_by_cpty, 
                              self.d_vanillas_on_fx_f32, self.d_vanillas_on_fx_i32,
                              self.d_vanillas_on_fx_b8, self.d_irs_f32,
                              self.d_irs_i32, self.d_zcs_f32, self.d_zcs_i32,
                              self.dt, self.max_coarse_per_reset, self.cDtoH_freq, True)
        self.stream.synchronize()
        self.d_mtm_by_cpty[0].copy_to_host(ary=self.mtm_by_cpty[0], stream=self.stream)
        self.d_cash_flows_by_cpty[0].copy_to_host(ary=self.cash_flows_by_cpty[0], stream=self.stream)
        
        _cuda_bulk_diffuse_event_begin = [cuda.event() for i in range(end)]
        _cuda_bulk_diffuse_event_end = [cuda.event() for i in range(end)]

        _cuda_compute_mtm_event_begin = [cuda.event() for i in range(end)]
        _cuda_compute_mtm_event_end = [cuda.event() for i in range(end)]

        _cuda_nested_cva_event_begin = [cuda.event() for i in range(end)]
        _cuda_nested_cva_event_end = [cuda.event() for i in range(end)]

        _cuda_nested_im_event_begin = [cuda.event() for i in range(end)]
        _cuda_nested_im_event_end = [cuda.event() for i in range(end)]

        for coarse_idx in range(1, end+1):
            t += self.dT
            idx_in_dev_arr = (coarse_idx-1) % self.cDtoH_freq + 1
            if not fused:
                _cuda_bulk_diffuse_event_begin[coarse_idx-1].record(stream=self.stream)
                self.cuda_bulk_diffuse(idx_in_dev_arr, t, self.d_X,
                                    self.d_def_indicators, self.d_dom_rate_integral,
                                    self.d_spread_integrals,
                                    self.d_irs_f32, self.d_irs_i32, 
                                    self.d_exp_1, self.d_rng_states,
                                    self.dt, self.max_coarse_per_reset)
                _cuda_bulk_diffuse_event_end[coarse_idx-1].record(stream=self.stream)
                _cuda_compute_mtm_event_begin[coarse_idx-1].record(stream=self.stream)
                self.cuda_compute_mtm(idx_in_dev_arr, t, self.d_X,
                                    self.d_mtm_by_cpty,
                                    self.d_cash_flows_by_cpty, 
                                    self.d_vanillas_on_fx_f32,
                                    self.d_vanillas_on_fx_i32,
                                    self.d_vanillas_on_fx_b8, self.d_irs_f32,
                                    self.d_irs_i32, self.d_zcs_f32, self.d_zcs_i32,
                                    self.dt, self.max_coarse_per_reset, self.cDtoH_freq, False)
                _cuda_compute_mtm_event_end[coarse_idx-1].record(stream=self.stream)
            else:
                _cuda_bulk_diffuse_event_begin[coarse_idx-1].record(stream=self.stream)
                if idx_in_dev_arr == 1:
                    self.cuda_diffuse_and_price(1, self.cDtoH_freq, t, self.d_X,
                                        self.d_dom_rate_integral,
                                        self.d_spread_integrals, self.d_mtm_by_cpty,
                                        self.d_cash_flows_by_cpty, 
                                        self.d_irs_f32, self.d_irs_i32, self.d_vanillas_on_fx_f32,
                                        self.d_vanillas_on_fx_i32, self.d_vanillas_on_fx_b8, 
                                        self.d_rng_states, self.dt, self.max_coarse_per_reset, 
                                        self.cDtoH_freq)
                    self.cuda_oversimulate_defs(1, self.cDtoH_freq, self.d_def_indicators, 
                                            self.d_spread_integrals, self.d_exp_1)
                _cuda_bulk_diffuse_event_end[coarse_idx-1].record(stream=self.stream)
            if nested_cva_at is not None:
                _cuda_nested_cva_event_begin[coarse_idx-1].record(stream=self.stream)
                if coarse_idx in nested_cva_at:
                    self.cuda_nested_cva(idx_in_dev_arr, self.num_coarse_steps-coarse_idx, t, self.d_X, self.d_def_indicators, self.d_dom_rate_integral, self.d_spread_integrals, self.d_mtm_by_cpty, self.d_cash_flows_by_cpty, self.d_irs_f32, self.d_irs_i32, self.d_vanillas_on_fx_f32, self.d_vanillas_on_fx_i32, self.d_vanillas_on_fx_b8, self.d_exp_1, self.d_rng_states, self.dt, self.cDtoH_freq, indicator_in_cva, self.d_nested_cva, self.d_nested_cva_sq)
                    self.d_nested_cva.copy_to_host(ary=self.nested_cva[coarse_idx], stream=self.stream)
                    self.d_nested_cva_sq.copy_to_host(ary=self.nested_cva_sq[coarse_idx], stream=self.stream)
                _cuda_nested_cva_event_end[coarse_idx-1].record(stream=self.stream)
            
            if nested_im_at is not None:
                _cuda_nested_im_event_begin[coarse_idx-1].record(stream=self.stream)
                if coarse_idx in nested_im_at:
                    for adam_iter in range(self.num_adam_iters):
                        adam_init = adam_iter == 0
                        step_size = self.lam * (adam_iter + 1)**(-self.gamma)
                        self.cuda_nested_im(alpha, adam_init, step_size, idx_in_dev_arr, im_window, t, self.d_X, self.d_mtm_by_cpty[idx_in_dev_arr], self.d_irs_f32, self.d_irs_i32, self.d_vanillas_on_fx_f32, self.d_vanillas_on_fx_i32, self.d_vanillas_on_fx_b8, self.d_rng_states, self.dt, self.d_nested_im_by_cpty, self.d_nested_im_std_by_cpty, self.d_nested_im_m, self.d_nested_im_v, self.adam_b1, self.adam_b2, adam_iter)
                    self.d_nested_im_by_cpty.copy_to_host(ary=self.nested_im_by_cpty[coarse_idx], stream=self.stream)
                _cuda_nested_im_event_end[coarse_idx-1].record(stream=self.stream)

            if coarse_idx % self.cDtoH_freq == 0:
                self.d_X[self.max_coarse_per_reset:].copy_to_host(
                    ary=self.X[coarse_idx-self.cDtoH_freq+1:coarse_idx+1], stream=self.stream)
                self.d_spread_integrals[1:].copy_to_host(
                    ary=self.spread_integrals[coarse_idx-self.cDtoH_freq+1:coarse_idx+1], stream=self.stream)
                self.d_dom_rate_integral[1:].copy_to_host(
                    ary=self.dom_rate_integral[coarse_idx-self.cDtoH_freq+1:coarse_idx+1], stream=self.stream)
                self.d_def_indicators[1:].copy_to_host(
                    ary=self.def_indicators[coarse_idx-self.cDtoH_freq+1:coarse_idx+1], stream=self.stream)
                self.d_mtm_by_cpty[1:].copy_to_host(
                    ary=self.mtm_by_cpty[coarse_idx-self.cDtoH_freq+1:coarse_idx+1], stream=self.stream)
                self.d_cash_flows_by_cpty[1:].copy_to_host(
                    ary=self.cash_flows_by_cpty[coarse_idx-self.cDtoH_freq+1:coarse_idx+1], stream=self.stream)
                self.d_X[:self.max_coarse_per_reset].copy_to_device(
                    self.d_X[-self.max_coarse_per_reset:], stream=self.stream)
                self.d_spread_integrals[0].copy_to_device(
                    self.d_spread_integrals[self.cDtoH_freq], stream=self.stream)
                self.d_dom_rate_integral[0].copy_to_device(
                    self.d_dom_rate_integral[self.cDtoH_freq], stream=self.stream)
                self.d_def_indicators[0].copy_to_device(
                    self.d_def_indicators[self.cDtoH_freq], stream=self.stream)

        if end % self.cDtoH_freq != 0:
            start_idx = (end // self.cDtoH_freq) * self.cDtoH_freq + 1
            length = end % self.cDtoH_freq
            self.d_X[self.max_coarse_per_reset:self.max_coarse_per_reset+length].copy_to_host(
                ary=self.X[start_idx:start_idx+length], stream=self.stream)
            self.d_spread_integrals[1:length+1].copy_to_host(
                ary=self.spread_integrals[start_idx:start_idx+length], stream=self.stream)
            self.d_dom_rate_integral[1:length+1].copy_to_host(
                ary=self.dom_rate_integral[start_idx:start_idx+length], stream=self.stream)
            self.d_def_indicators[1:length+1].copy_to_host(
                ary=self.def_indicators[start_idx:start_idx+length], stream=self.stream)
            self.d_mtm_by_cpty[1:length+1].copy_to_host(
                ary=self.mtm_by_cpty[start_idx:start_idx+length], stream=self.stream)
            self.d_cash_flows_by_cpty[1:length+1].copy_to_host(
                ary=self.cash_flows_by_cpty[start_idx:start_idx+length], stream=self.stream)

        if verbose:
            print('Everything was successfully queued!')
        
        for evt_cuda_bulk_diffuse_event, evt_cuda_compute_mtm_event, evt_cuda_nested_cva_event, evt_cuda_nested_im_event in zip(_cuda_bulk_diffuse_event_end, _cuda_compute_mtm_event_end, _cuda_nested_cva_event_end, _cuda_nested_im_event_end):
            evt_cuda_bulk_diffuse_event.synchronize()
            evt_cuda_compute_mtm_event.synchronize()
            evt_cuda_nested_cva_event.synchronize()
            evt_cuda_nested_im_event.synchronize()
        
        self.stream.synchronize()
        
        if not fused:
            print('cuda_bulk_diffuse average elapsed time per launch: {0} ms'.format(round(sum(cuda.event_elapsed_time(evt_begin, evt_end) for evt_begin, evt_end in zip(_cuda_bulk_diffuse_event_begin, _cuda_bulk_diffuse_event_end))/end, 3)))
            print('compute_mtm average elapsed time per launch: {0} ms'.format(round(sum(cuda.event_elapsed_time(evt_begin, evt_end) for evt_begin, evt_end in zip(_cuda_compute_mtm_event_begin, _cuda_compute_mtm_event_end))/end, 3)))
        else:
            print('cuda_diffuse_and_price elapsed time: {0} ms'.format(round(sum(cuda.event_elapsed_time(evt_begin, evt_end) for evt_begin, evt_end in zip(_cuda_bulk_diffuse_event_begin, _cuda_bulk_diffuse_event_end)), 3)))
        
        if nested_cva_at is not None:
            print('cuda_nested_cva average elapsed time per launch: {0} ms'.format(round(sum(cuda.event_elapsed_time(evt_begin, evt_end) for evt_begin, evt_end in zip(_cuda_nested_cva_event_begin, _cuda_nested_cva_event_end))/len(nested_cva_at), 3)))
        
        if nested_im_at is not None:
            print('cuda_nested_im average elapsed time per launch: {0} ms'.format(round(sum(cuda.event_elapsed_time(evt_begin, evt_end) for evt_begin, evt_end in zip(_cuda_nested_im_event_begin, _cuda_nested_im_event_end))/len(nested_im_at), 3)))

        # TODO: port this to CUDA
        self.cash_pos_by_cpty = ne.evaluate(
            'c*exp(-r)', 
            local_dict={
                'c': self.cash_flows_by_cpty,
                'r': self.dom_rate_integral[:, None, :]
            }
        )
        np.cumsum(self.cash_pos_by_cpty, axis=0, out=self.cash_pos_by_cpty)
        self.cash_pos_by_cpty *= np.exp(self.dom_rate_integral[:, None, :])
    
    def reset_rng_states(self):
        self.d_rng_states = create_xoroshiro128p_states(self.num_paths*(self.num_defs_per_path+self.num_inner_paths), seed=1)
