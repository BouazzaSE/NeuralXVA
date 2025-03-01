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
from simulation.kernels import compile_cuda_compute_mtm, compile_cuda_diffuse_and_price, compile_cuda_oversimulate_defs, compile_cuda_generate_exp1, compile_cuda_nested_cva, compile_cuda_nested_im, compile_cuda_nested_im_err

class DiffusionEngine:
    def __init__(self, irs_batch_size, vanilla_batch_size, num_coarse_steps, dT, num_fine_per_coarse, dt, num_paths, num_inner_paths, 
                 num_defs_per_path, num_rates, num_spreads, R, rates_params, fx_params,
                 spreads_params, vanilla_specs, irs_specs, zcs_specs,
                 initial_values, initial_defaults, cDtoH_freq, device=0, params_in_const=True, no_nested_cva=False, no_nested_im=False, num_adam_iters=100, lam=1, gamma=0.5, adam_b1=0.9, adam_b2=0.999):
        cuda.select_device(device)
        self.params_in_const = params_in_const  # True: model parameters are put in constant memory, false: they are put in global memory instead
        self.irs_batch_size = irs_batch_size    # size of the batch of swaps to be loaded in shared memory (shared memory is used as a buffer for product specs during MtM computations)
        self.vanilla_batch_size = vanilla_batch_size    # # size of the batch of vanill options to be loaded in shared memory (shared memory is used as a buffer for product specs during MtM computations)
        self.num_coarse_steps = num_coarse_steps    # number of coarse time steps
        self.num_fine_per_coarse = num_fine_per_coarse  # number of fine time steps per each coarse time step
        self.num_paths = num_paths  # number of outer diffusion paths
        self.num_inner_paths = num_inner_paths  # number of inner diffusion paths for the NMC procedures
        self.num_defs_per_path = num_defs_per_path  # number of default simulations conditional on each diffusion path
        self.num_rates = num_rates  # number of short rates ( = number of currencies, since 1 short rate = 1 currency)
        self.num_spreads = num_spreads  # number of spreads ( = 1 + number of counterparties, since the first spread is always that of the bank)
        self.vanilla_specs = vanilla_specs.copy()   # named array containing specifications of the vanilla options to be priced, each row corresponds to one vanilla option
        self.irs_specs = irs_specs.copy()   # named array containing specifications of the swaps to be priced, each row corresponds to one swap
        self.zcs_specs = zcs_specs.copy()   # NOT USED (TODO: à nettoyer et à enlever)
        self.cDtoH_freq = cDtoH_freq    # size in coarse steps of the path to be simulated on GPU (we simulate the paths by time slices because of memory constraints)
        self.no_nested_cva = no_nested_cva  # True: no kernel compilation & no allocations are to be done for the nested CVA, False: kernel & memory space will be prepared for the nested CVA
        self.no_nested_im = no_nested_im    # True: no kernel compilation & no allocations are to be done for the nested IM, False: kernel & memory space will be prepared for the nested IM
        self.num_adam_iters = num_adam_iters    # number of Adam iterations for the nested stochastic approximation of the IM
        self.lam = lam
        self.gamma = gamma  # Adam step size for the nested stochastic approximation of the IM
        self.adam_b1 = adam_b1  # exponential moving average parameter for the 1st moment of gradient in the Adam algorithm for the nested stochastic approximation of the IM
        self.adam_b2 = adam_b2  # exponential moving average parameter for the 2nd moment of gradient in the Adam algorithm for the nested stochastic approximation of the IM
        
        if irs_specs.size > 0:
            self.max_coarse_per_reset = max(int((self.irs_specs['reset_freq'].max()+dt)/dT), 1)
        else:
            self.max_coarse_per_reset = 1
        # TODO: add assert statements on the acceptable range for reset_freq

        # force casting of float constants
        self.dt = np.float32(dt)
        self.dT = np.float32(dT)

        # total number of diffusion factors (short rates + FX + spreads)
        self.num_diffusions = 2*self.num_rates+self.num_spreads-1

        # CUDA stream to have asynchronous kernel launches & copies to hide the latencies associated with those calls
        self.stream = cuda.stream()

        # preparing workspace arrays on the host and the device
        self._allocate_host_arrays()
        self._set_cpu_arrays(R, rates_params, fx_params, spreads_params,
                             initial_values, initial_defaults)
        self._allocate_device_arrays()
        self._copy_product_specs_to_device()

        # running factories which will generate custom CUDA kernels optimized for our problem size
        self.cuda_generate_exp1 = compile_cuda_generate_exp1(self.num_spreads,
                                                             self.num_defs_per_path,
                                                             self.num_paths,
                                                             512,
                                                             self.stream)
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
                                                         self.stream, params_in_const=params_in_const)
        self.cuda_oversimulate_defs = compile_cuda_oversimulate_defs(self.num_spreads,
                                                         self.num_defs_per_path,
                                                         self.num_paths, 
                                                         512,
                                                         self.stream)
        if not self.no_nested_cva:
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
        if not self.no_nested_im:
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
            self.cuda_nested_im_err = compile_cuda_nested_im_err(self.irs_batch_size, 
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
        # creating RNG state structures on the GPU
        self.d_rng_states = None
        self.reset_rng_states()

    def _allocate_host_arrays(self):
        # CPU array for the diffusion factors
        self.X = cuda.pinned_array(
            (self.num_coarse_steps+1, self.num_diffusions, self.num_paths), np.float32)
        # CPU array for the MtMs for each counterparty
        self.mtm_by_cpty = cuda.pinned_array(
            (self.num_coarse_steps+1, self.num_spreads-1, self.num_paths), np.float32)
        # CPU array for the cash flows for each counterparty
        self.cash_flows_by_cpty = cuda.pinned_array(
            (self.num_coarse_steps+1, self.num_spreads-1, self.num_paths), np.float32)
        # CPU array for the cash position (ie accumulation of the cash flows) for each counterparty
        self.cash_pos_by_cpty = cuda.pinned_array(
            (self.num_coarse_steps+1, self.num_spreads-1, self.num_paths), np.float32)
        # CPU array for the spread integrals
        self.spread_integrals = cuda.pinned_array(
            (self.num_coarse_steps+1, self.num_spreads, self.num_paths), np.float32)
        # CPU array for the domestic short rate integral
        self.dom_rate_integral = cuda.pinned_array(
            (self.num_coarse_steps+1, self.num_paths), np.float32)
        # CPU array for the default indicators
        self.def_indicators = cuda.pinned_array(
            (self.num_coarse_steps+1, (self.num_spreads-1+7)//8, self.num_defs_per_path, self.num_paths), 
            np.int8)
        # CPU array for the nested CVA
        if not self.no_nested_cva:
            # since the CPU array for the nested CVA can be huge (mostly due to the fact what we have an additional dimension related to the default scenario)
            # we first try to allocate it in pinned memory, and if it fails, we allocate it
            # using the regular numpy allocator
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
        
        # CPU array for the nested IM, same remarks as for the CVA
        if not self.no_nested_im:
            try:
                self.nested_im_by_cpty = cuda.pinned_array(
                    (self.num_coarse_steps+1, self.num_spreads-1, self.num_paths), np.float32)
            except cuda.cudadrv.driver.CudaAPIError:
                print('couldn\'t allocate pinned array for nested_im_by_cpty, using the numpy allocator instead (non-pinned array).')
                self.nested_im_by_cpty = np.empty((self.num_coarse_steps+1, self.num_spreads-1, self.num_paths), np.float32)
            try:
                self.nested_im_err_by_cpty = cuda.pinned_array(
                    (self.num_coarse_steps+1, self.num_spreads-1, self.num_paths), np.float32)
            except cuda.cudadrv.driver.CudaAPIError:
                print('couldn\'t allocate pinned array for nested_im_err_by_cpty, using the numpy allocator instead (non-pinned array).')
                self.nested_im_err_by_cpty = np.empty((self.num_coarse_steps+1, self.num_spreads-1, self.num_paths), np.float32)

        # correlation matrix for the Brownian motions
        self.R = np.empty(
            (self.num_diffusions, self.num_diffusions), dtype=np.float32)
        # then we store only upper-triangular entries of the correlation matrix and its Cholesky decomposition
        # the following matrices are flattened
        self.g_R = np.empty(self.num_diffusions * (self.num_diffusions+1)//2, np.float32)
        self.g_L_T = np.empty(self.num_diffusions * (self.num_diffusions+1)//2, np.float32)

        self.g_diff_params = np.empty(5*self.num_rates-2+3*self.num_spreads, np.float32)

        # grouping product specs by data type
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
        # same as _allocate_host_arrays but on GPU
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
            self.d_nested_im_err_by_cpty = cuda.device_array((self.num_spreads-1, self.num_paths), np.float32)
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
        self.d_cash_pos_by_cpty = cuda.device_array(
            (self.cDtoH_freq+1, self.num_spreads-1, self.num_paths), np.float32)
    
    def _set_cpu_arrays(self, R, rates_params, fx_params, spreads_params,
                        initial_values, initial_defaults):
        assert R.shape[0] == R.shape[1] == self.num_diffusions, \
            'incorrect shape for correlation matrix'
        assert R.dtype == np.float32, 'use only float32 for floating point numbers'
        # setting the CPU arrays for the correlation matrix, its upper-diagonal entries and those of its Cholesky decomposition
        self.R[:] = R
        triu_indices = np.triu_indices(self.num_diffusions)
        self.g_R[:] = self.R[triu_indices]
        self.g_L_T[:] = np.linalg.cholesky(self.R).T[triu_indices]

        # setting the CPU array for the diffusion parameters with the Vasicek parameters
        self.g_diff_params[:self.num_rates] = rates_params['a']
        self.g_diff_params[self.num_rates:2*self.num_rates] = rates_params['b']
        self.g_diff_params[2*self.num_rates:3 *
                           self.num_rates] = rates_params['sigma']
        # setting the CPU array for the diffusion parameters with the FX parameters (BS volatility)
        self.g_diff_params[3*self.num_rates:4 *
                           self.num_rates-1] = fx_params['vol']
        # what is called here "drift adjustment" is basically the change of probability adjustment 
        # that needs to be done to the rate diffusions to ensure everything is diffused under the 
        # same risk-neutral measure
        # BEGIN DRIFT ADJ
        self.g_diff_params[4*self.num_rates-1:5*self.num_rates-2] = -self.g_diff_params[3*self.num_rates:4 *
                                                                                        self.num_rates-1]*self.R[1:self.num_rates, self.num_rates:(2*self.num_rates-1)].diagonal()
        # END DRIFT ADJ
        # setting the CPU array for the diffusion paramters with the CIR parameters
        self.g_diff_params[5*self.num_rates-2:5*self.num_rates +
                           self.num_spreads-2] = spreads_params['a']
        self.g_diff_params[5*self.num_rates+self.num_spreads-2:5 *
                           self.num_rates+2*self.num_spreads-2] = spreads_params['b']
        self.g_diff_params[5*self.num_rates+2 *
                           self.num_spreads-2:] = spreads_params['vvol']

        # setting the CPU arrays for the vanilla specs
        self.vanillas_on_fx_f32[:, 0] = self.vanilla_specs['maturity']
        self.vanillas_on_fx_f32[:, 1] = self.vanilla_specs['notional']
        self.vanillas_on_fx_f32[:, 2] = self.vanilla_specs['strike']
        self.vanillas_on_fx_i32[:, 0] = self.vanilla_specs['cpty']
        self.vanillas_on_fx_i32[:, 1] = self.vanilla_specs['undl']
        self.vanillas_on_fx_b8[:, 0] = self.vanilla_specs['call_put']

        # setting the CPU arrays for the swap specs
        self.irs_f32[:, 0] = self.irs_specs['first_reset']
        self.irs_f32[:, 1] = self.irs_specs['reset_freq']
        self.irs_f32[:, 2] = self.irs_specs['notional']
        self.irs_f32[:, 3] = self.irs_specs['swap_rate']
        self.irs_i32[:, 0] = self.irs_specs['num_resets']
        self.irs_i32[:, 1] = self.irs_specs['cpty']
        self.irs_i32[:, 2] = self.irs_specs['undl']

        # setting the CPU arrays for the ZCs specs (UNUSED, DON'T ATTEMPT TO USE)
        # TODO: remove them altogether
        self.zcs_f32[:, 0] = self.zcs_specs['maturity']
        self.zcs_f32[:, 1] = self.zcs_specs['notional']
        self.zcs_i32[:, 0] = self.zcs_specs['cpty']
        self.zcs_i32[:, 1] = self.zcs_specs['undl']

        # initializing integrals at 0 at time 0
        self.spread_integrals[0] = 0.
        self.dom_rate_integral[0] = 0.

        # initializing the default states
        self.def_indicators[0] = initial_defaults[:, np.newaxis, np.newaxis]

        # setting the initial values for the risk factors
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

    def generate_batch(self, end=None, verbose=False, fused=False, nested_cva_at=None, nested_im_at=None, indicator_in_cva=False, alpha=None, im_window=None, set_irs_at_par=True):
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
                              self.dt, self.max_coarse_per_reset, self.cDtoH_freq, set_irs_at_par)
        self.stream.synchronize()
        self.d_mtm_by_cpty[0].copy_to_host(ary=self.mtm_by_cpty[0], stream=self.stream)
        self.d_cash_flows_by_cpty[0].copy_to_host(ary=self.cash_flows_by_cpty[0], stream=self.stream)
        self.d_cash_pos_by_cpty[0].copy_to_device(self.d_cash_flows_by_cpty[0], stream=self.stream)
        self.cash_pos_by_cpty[0] = self.cash_flows_by_cpty[0]
        
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
                raise NotImplementedError
                # _cuda_bulk_diffuse_event_begin[coarse_idx-1].record(stream=self.stream)
                # self.cuda_bulk_diffuse(idx_in_dev_arr, t, self.d_X,
                #                     self.d_def_indicators, self.d_dom_rate_integral,
                #                     self.d_spread_integrals,
                #                     self.d_irs_f32, self.d_irs_i32, 
                #                     self.d_exp_1, self.d_rng_states,
                #                     self.dt, self.max_coarse_per_reset)
                # _cuda_bulk_diffuse_event_end[coarse_idx-1].record(stream=self.stream)
                # _cuda_compute_mtm_event_begin[coarse_idx-1].record(stream=self.stream)
                # self.cuda_compute_mtm(idx_in_dev_arr, t, self.d_X,
                #                     self.d_mtm_by_cpty,
                #                     self.d_cash_flows_by_cpty, 
                #                     self.d_vanillas_on_fx_f32,
                #                     self.d_vanillas_on_fx_i32,
                #                     self.d_vanillas_on_fx_b8, self.d_irs_f32,
                #                     self.d_irs_i32, self.d_zcs_f32, self.d_zcs_i32,
                #                     self.dt, self.max_coarse_per_reset, self.cDtoH_freq, False)
                # _cuda_compute_mtm_event_end[coarse_idx-1].record(stream=self.stream)
            else:
                _cuda_bulk_diffuse_event_begin[coarse_idx-1].record(stream=self.stream)
                if idx_in_dev_arr == 1:
                    self.cuda_diffuse_and_price(1, self.cDtoH_freq, t, self.d_X,
                                        self.d_dom_rate_integral,
                                        self.d_spread_integrals, self.d_mtm_by_cpty,
                                        self.d_cash_flows_by_cpty, 
                                        self.d_cash_pos_by_cpty, 
                                        self.d_irs_f32, self.d_irs_i32, self.d_vanillas_on_fx_f32,
                                        self.d_vanillas_on_fx_i32, self.d_vanillas_on_fx_b8, 
                                        self.d_rng_states, self.dt, self.max_coarse_per_reset, 
                                        self.g_diff_params, self.g_R, self.g_L_T)
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
                    self.cuda_nested_im_err(alpha, idx_in_dev_arr, im_window, t, self.d_X, self.d_mtm_by_cpty[idx_in_dev_arr], self.d_irs_f32, self.d_irs_i32, self.d_vanillas_on_fx_f32, self.d_vanillas_on_fx_i32, self.d_vanillas_on_fx_b8, self.d_rng_states, self.dt, self.d_nested_im_by_cpty, self.d_nested_im_err_by_cpty)
                    self.d_nested_im_err_by_cpty.copy_to_host(ary=self.nested_im_err_by_cpty[coarse_idx], stream=self.stream)
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
                self.d_cash_pos_by_cpty[1:].copy_to_host(
                    ary=self.cash_pos_by_cpty[coarse_idx-self.cDtoH_freq+1:coarse_idx+1], stream=self.stream)
                if coarse_idx < end:
                    self.d_X[:self.max_coarse_per_reset].copy_to_device(
                        self.d_X[-self.max_coarse_per_reset:], stream=self.stream)
                    self.d_spread_integrals[0].copy_to_device(
                        self.d_spread_integrals[self.cDtoH_freq], stream=self.stream)
                    self.d_dom_rate_integral[0].copy_to_device(
                        self.d_dom_rate_integral[self.cDtoH_freq], stream=self.stream)
                    self.d_def_indicators[0].copy_to_device(
                        self.d_def_indicators[self.cDtoH_freq], stream=self.stream)
                    self.d_cash_pos_by_cpty[0].copy_to_device(
                        self.d_cash_pos_by_cpty[self.cDtoH_freq], stream=self.stream)

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
            self.d_cash_pos_by_cpty[1:length+1].copy_to_host(
                ary=self.cash_pos_by_cpty[start_idx:start_idx+length], stream=self.stream)

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
    
    def single_step_diffuse_and_price(self, coarse_idx):
        t = (coarse_idx+1)*self.dT
        padding = max(self.max_coarse_per_reset-1-coarse_idx, 0)
        self.d_X[padding:self.max_coarse_per_reset].copy_to_device(
            ary=self.X[max(coarse_idx-self.max_coarse_per_reset+1, 0):coarse_idx+1], stream=self.stream
        )
        self.d_spread_integrals[0].copy_to_device(
            ary=self.spread_integrals[coarse_idx], stream=self.stream
        )
        self.d_dom_rate_integral[0].copy_to_device(
            ary=self.dom_rate_integral[coarse_idx], stream=self.stream
        )
        self.d_def_indicators[0].copy_to_device(
            ary=self.def_indicators[coarse_idx], stream=self.stream
        )
        self.d_mtm_by_cpty[0].copy_to_device(
            ary=self.mtm_by_cpty[coarse_idx], stream=self.stream
        )
        self.d_cash_flows_by_cpty[0].copy_to_device(
            ary=self.cash_flows_by_cpty[coarse_idx], stream=self.stream
        )
        self.d_cash_pos_by_cpty[0].copy_to_device(
            ary=self.cash_pos_by_cpty[coarse_idx], stream=self.stream
        )
        self.cuda_diffuse_and_price(1, 1, t, self.d_X,
                                        self.d_dom_rate_integral,
                                        self.d_spread_integrals, self.d_mtm_by_cpty,
                                        self.d_cash_flows_by_cpty, 
                                        self.d_cash_pos_by_cpty, 
                                        self.d_irs_f32, self.d_irs_i32, self.d_vanillas_on_fx_f32,
                                        self.d_vanillas_on_fx_i32, self.d_vanillas_on_fx_b8, 
                                        self.d_rng_states, self.dt, self.max_coarse_per_reset, 
                                        self.g_diff_params, self.g_R, self.g_L_T)
        self.d_def_indicators[1] = self.d_def_indicators[0]
        self.cuda_oversimulate_defs(1, 1, self.d_def_indicators, 
                                    self.d_spread_integrals, self.d_exp_1)
    
    def reset_rng_states(self):
        self.d_rng_states = create_xoroshiro128p_states(self.num_paths*(self.num_defs_per_path+self.num_inner_paths), seed=1)
