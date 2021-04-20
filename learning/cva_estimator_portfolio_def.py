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

from learning.generic_estimator import GenericEstimator
from learning.xva_estimator import XVAEstimatorPortfolio
import math
import numba as nb
from numba import cuda
import numpy as np
import time
import torch


def compile_cuda_build_labels_backward(num_spreads, num_defs_per_path, num_paths, ntpb, stream):
    sig = (nb.int8[:, :, :], nb.int8[:, :, :], nb.float32[:], nb.float32[:], nb.float32[:, :], nb.float32[:, :], nb.bool_, nb.bool_)

    @cuda.jit(func_or_sig=sig, max_registers=32)
    def _build_labels_backward(def_now, def_next, rate_integral_now, rate_integral_next, mtm_next, out, implicit_timestepping, accumulate):
        block = cuda.blockIdx.x
        block_size = cuda.blockDim.x
        tidx = cuda.threadIdx.x
        pos = tidx + block * block_size
        # TODO: work on a local array instead of out
        if pos < num_paths:
            if not accumulate:
                for j in range(num_defs_per_path):
                    out[j, pos] = 0
            for cpty in range(num_spreads-1):
                m = mtm_next[cpty, pos]
                if m < 0:
                    m = 0
                q = cpty // 8
                r = cpty % 8
                for j in range(num_defs_per_path):
                    di = (def_next[q, j, pos] ^ def_now[q, j, pos]) & def_next[q, j, pos]
                    if di & (1 << r):
                        out[j, pos] += m
            df_r = math.exp(rate_integral_now[pos] - rate_integral_next[pos])
            for j in range(num_defs_per_path):
                out[j, pos] *= df_r
            if implicit_timestepping:
                for q in range((num_spreads+6)//8):
                    for j in range(num_defs_per_path):
                        def_next[q, j, pos] = def_now[q, j, pos]
                rate_integral_next[pos] = rate_integral_now[pos]
    
    _build_labels_backward._func.get().cache_config(prefer_cache=True)
    build_labels_backward = _build_labels_backward[(num_paths+ntpb-1)//ntpb, ntpb, stream]
    
    return build_labels_backward

def compile_cuda_aggregate_survival(num_spreads, num_defs_per_path, num_paths, ntpb, stream):
    sig = (nb.float32[:, :, :], nb.int8[:, :, :], nb.float32[:, :])

    @cuda.jit(func_or_sig=sig, max_registers=32)
    def _aggregate_survival(labels, def_arr, out):
        block = cuda.blockIdx.x
        block_size = cuda.blockDim.x
        tidx = cuda.threadIdx.x
        pos = tidx + block * block_size

        if pos < num_paths:
            for i in range(num_defs_per_path):
                out[i, pos] = 0
            for cpty in range(num_spreads-1):
                q = cpty // 8
                r = cpty % 8
                mask = 1 << r
                for i in range(num_defs_per_path):
                    di = def_arr[q, i, pos] & mask
                    if not di:
                        out[i, pos] += labels[cpty, i, pos]
                        
    _aggregate_survival._func.get().cache_config(prefer_cache=True)
    aggregate_survival = _aggregate_survival[(num_paths+ntpb-1)//ntpb, ntpb, stream]
    
    return aggregate_survival

def compile_unpack(num_spreads):
    @nb.jit((nb.int8[:, :, :], nb.float32[:, :, :]), nopython=True, nogil=True)
    def _unpack(def_arr, out):
        for cpty in range(num_spreads-1):
            out[:, :, cpty] = def_arr[cpty//8] & (1 << (cpty%8))
    return _unpack

_cuda_build_labels_backward_cache = {}
_cuda_aggregate_survival_cache = {}
_unpack_cache = {}

class CVAEstimatorPortfolioDef(XVAEstimatorPortfolio):
    def __init__(self, prev_reset_arr, backward, warmup, compute_loss_surface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features = 3*self.diffusion_engine.num_rates+2*(self.diffusion_engine.num_spreads-1)-1
        self.backward = backward
        self.warmup = warmup
        regr_type = 'positive_mean' if not self.linear else 'mean'
        self._estimator = GenericEstimator(self.num_features, self.num_hidden_layers, \
            self.num_hidden_units, self.diffusion_engine.num_defs_per_path*self.diffusion_engine.num_paths, \
                self.batch_size, self.num_epochs, self.lr, self.holdout_size, self.device, \
                    regr_type=regr_type, linear=self.linear, best_sol=self.best_sol)
        self.saved_states = [None] * (self.diffusion_engine.num_coarse_steps+1)
        self.prev_reset_arr = prev_reset_arr
        self.compute_loss_surface = compute_loss_surface
        if compute_loss_surface:
            self._loss_surface = np.empty((self.diffusion_engine.num_coarse_steps+1, self.num_epochs), np.float32)
        self._compile_kernels()
    
    def _compile_kernels(self):
        # NOTE: not caring about async kernel launches for now
        self.__cuda_build_labels_backward = _cuda_build_labels_backward_cache.get((self.diffusion_engine.num_paths, 512, 0))
        self.__cuda_aggregate_survival = _cuda_aggregate_survival_cache.get((self.diffusion_engine.num_paths, 512, 0))
        if self.backward:
            if self.__cuda_build_labels_backward is None:
                self.__cuda_build_labels_backward = compile_cuda_build_labels_backward(self.diffusion_engine.num_spreads, self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths, 512, 0)
                _cuda_build_labels_backward_cache[(self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths, 512, 0)] = self.__cuda_build_labels_backward
        else:
            raise NotImplementedError
        if self.__cuda_aggregate_survival is None:
            self.__cuda_aggregate_survival = compile_cuda_aggregate_survival(self.diffusion_engine.num_spreads, self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths, 512, 0)
            _cuda_aggregate_survival_cache[(self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths, 512, 0)] = self.__cuda_aggregate_survival
        self.__unpack = _unpack_cache.get(self.diffusion_engine.num_spreads)
        if self.__unpack is None:
            self.__unpack = compile_unpack(self.diffusion_engine.num_spreads)
            _unpack_cache[self.diffusion_engine.num_spreads] = self.__unpack
    
    def _build_features(self):
        features = cuda.pinned_array((self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths, self.num_features), dtype=np.float32)
        while True:
            t = yield
            t_prev_reset = self.prev_reset_arr[t]
            features[:, :, :2*self.diffusion_engine.num_rates-1] = self.diffusion_engine.X[t, :2*self.diffusion_engine.num_rates-1].T[None]
            np.maximum(
                self.diffusion_engine.X[t, 2*self.diffusion_engine.num_rates:2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-1].T[None], 
                0., 
                out=features[:, :, 2*self.diffusion_engine.num_rates-1:2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2]
            )
            if t_prev_reset > 0:
                features[:, :, 2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2] = self.diffusion_engine.X[t_prev_reset, :self.diffusion_engine.num_rates].T[None]
            else:
                features[:, :, 2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2] = 0
            self.__unpack(self.diffusion_engine.def_indicators[t], features[:, :, 3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:])
            yield features.reshape(self.diffusion_engine.num_defs_per_path*self.diffusion_engine.num_paths, -1)

    def _build_labels(self, as_cuda_tensor=False):
        if self.backward:
            return self._build_labels_backward(as_cuda_tensor)
        else:
            raise NotImplementedError
    
    def _build_labels_backward(self, as_cuda_tensor):
        d_def_now = self.diffusion_engine.d_def_indicators[0]
        d_def_next = self.diffusion_engine.d_def_indicators[1]
        d_mtm_next = self.diffusion_engine.d_mtm_by_cpty[0]
        d_rate_integral_now = self.diffusion_engine.d_dom_rate_integral[0]
        d_rate_integral_next = self.diffusion_engine.d_dom_rate_integral[1]
        t_out = torch.empty((self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths), dtype=torch.float32, device=self.device)
        with cuda.devices.gpus[self.device.index]:
            d_out = cuda.as_cuda_array(t_out)
        if as_cuda_tensor:
            out = t_out
        else:
            out = cuda.pinned_array((self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths), dtype=np.float32)
        out[:] = 0
        if as_cuda_tensor:
            yield out.view(-1, 1)
        else:
            yield out.reshape(-1, 1)
        d_def_next.copy_to_device(self.diffusion_engine.def_indicators[self.diffusion_engine.num_coarse_steps])
        d_rate_integral_next.copy_to_device(self.diffusion_engine.dom_rate_integral[self.diffusion_engine.num_coarse_steps])
        accumulate = False
        for t in range(self.diffusion_engine.num_coarse_steps-1, -1, -1):
            d_def_now.copy_to_device(self.diffusion_engine.def_indicators[t])
            d_rate_integral_now.copy_to_device(self.diffusion_engine.dom_rate_integral[t])
            d_mtm_next.copy_to_device(self.diffusion_engine.mtm_by_cpty[t+1])
            self.__cuda_build_labels_backward(d_def_now, d_def_next, d_rate_integral_now, d_rate_integral_next, d_mtm_next, d_out, t > 0, accumulate)
            if as_cuda_tensor:
                yield out.view(-1, 1)
            else:
                d_out.copy_to_host(out)
                yield out.reshape(-1, 1)
            if not accumulate:
                accumulate = True
