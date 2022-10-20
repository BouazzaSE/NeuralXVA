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


def compile_cuda_build_labels_backward(num_spreads, num_paths, ntpb, stream):
    sig = (nb.float32[:, :], nb.float32[:, :], nb.float32[:], nb.float32[:], nb.float32[:, :], nb.float32[:, :], nb.bool_, nb.bool_)

    @cuda.jit(func_or_sig=sig, max_registers=32)
    def _build_labels_backward(spread_integral_now, spread_integral_next, rate_integral_now, rate_integral_next, mtm_next, out, implicit_timestepping, accumulate):
        block = cuda.blockIdx.x
        block_size = cuda.blockDim.x
        tidx = cuda.threadIdx.x
        pos = tidx + block * block_size
        if pos < num_paths:
            dr = rate_integral_now[pos] - rate_integral_next[pos]
            df_r = math.exp(dr)
            for cpty in range(num_spreads-1):
                df_r_d = math.exp(dr+spread_integral_now[cpty, pos]-spread_integral_next[cpty, pos])
                m = mtm_next[cpty, pos]
                if m < 0:
                    m = 0
                if not accumulate:
                    out[cpty, pos] = m * (df_r - df_r_d)
                else:
                    out[cpty, pos] *= df_r_d
                    out[cpty, pos] += m * (df_r - df_r_d)
                if implicit_timestepping:
                    spread_integral_next[cpty, pos] = spread_integral_now[cpty, pos]
            if implicit_timestepping:
                rate_integral_next[pos] = rate_integral_now[pos]
    
    build_labels_backward = _build_labels_backward[(num_paths+ntpb-1)//ntpb, ntpb, stream]
    
    return build_labels_backward

def compile_cuda_aggregate_survival(num_spreads, num_defs_per_path, num_paths, ntpb, stream):
    sig = (nb.float32[:, :], nb.int8[:, :, :], nb.float32[:, :])

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
                        out[i, pos] += labels[cpty, pos]
    
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

class CVAEstimatorPortfolioInt(XVAEstimatorPortfolio):
    def __init__(self, prev_reset_arr, backward, warmup, compute_loss_surface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features = 3*self.diffusion_engine.num_rates+2*(self.diffusion_engine.num_spreads-1)-1
        self.backward = backward
        self.warmup = warmup
        regr_type = 'positive_mean' if not self.linear else 'mean'
        self._estimator = GenericEstimator(self.num_features, self.num_hidden_layers, \
            self.num_hidden_units, self.diffusion_engine.num_defs_per_path*self.diffusion_engine.num_paths, \
                self.batch_size, self.num_epochs, self.lr, self.holdout_size, self.device, \
                    regr_type=regr_type, linear=self.linear, best_sol=self.best_sol, refine_last_layer=self.refine_last_layer)
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
                self.__cuda_build_labels_backward = compile_cuda_build_labels_backward(self.diffusion_engine.num_spreads, self.diffusion_engine.num_paths, 512, 0)
                _cuda_build_labels_backward_cache[(self.diffusion_engine.num_paths, 512, 0)] = self.__cuda_build_labels_backward
        else:
            raise NotImplementedError
        if self.__cuda_aggregate_survival is None:
            self.__cuda_aggregate_survival = compile_cuda_aggregate_survival(self.diffusion_engine.num_spreads, self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths, 512, 0)
            _cuda_aggregate_survival_cache[(self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths, 512, 0)] = self.__cuda_aggregate_survival
        self.__unpack = _unpack_cache.get(self.diffusion_engine.num_spreads)
        if self.__unpack is None:
            self.__unpack = compile_unpack(self.diffusion_engine.num_spreads)
            _unpack_cache[self.diffusion_engine.num_spreads] = self.__unpack
    
    def _batch_generator(self, labels_as_cuda_tensors=True, train_mode=False):
        assert isinstance(self.batch_size, int) and (self.batch_size >= 1) and (not (self.batch_size & (self.batch_size-1)))
        features_gen = self._features_generator()
        labels_gpu = torch.empty(self.batch_size, 1, dtype=torch.float32, device=self.device)
        labels_gen = self._build_labels(labels_as_cuda_tensors)
        num_defs_per_batch = (self.batch_size+self.diffusion_engine.num_paths-1)//self.diffusion_engine.num_paths
        batch_size = min(self.batch_size, self.diffusion_engine.num_paths)
        if self.backward:
            timesteps = range(self.diffusion_engine.num_coarse_steps, -1, -1)
        else:
            timesteps = range(self.diffusion_engine.num_coarse_steps+1)
        for t in timesteps:
            next(features_gen)
            __gen_features = features_gen.send(t)
            labels = next(labels_gen).view(self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths, 1)
            def __gen_labels(mean=None, std=None):
                nonlocal labels_gpu
                for i in range((self.diffusion_engine.num_paths+batch_size-1)//batch_size):
                    for j in range((self.diffusion_engine.num_defs_per_path+num_defs_per_batch-1)//num_defs_per_batch):
                        labels_gpu.copy_(labels[j*num_defs_per_batch: (j+1)*num_defs_per_batch, i*batch_size:(i+1)*batch_size].view(-1, 1))
                        if mean is not None:
                            labels_gpu -= mean[None]
                        if std is not None:
                            labels_gpu /= (std[None] + 1e-7)
                        yield labels_gpu
            yield t, __gen_features, __gen_labels
    
    # def _features_generator(self):
    #     assert isinstance(self.batch_size, int) and (self.batch_size >= 1) and (not (self.batch_size & (self.batch_size-1)))
    #     num_cpty = self.diffusion_engine.num_spreads-1
    #     features_gpu = torch.empty(self.batch_size, self.num_features, dtype=torch.float32, device=self.device)
    #     def_indicators_gpu = torch.empty(self.batch_size, (num_cpty+7)//8, dtype=torch.uint8, device=self.device)
    #     _cpty_idx = np.arange(num_cpty, dtype=np.int32)
    #     _cpty_mask = torch.tensor(1 << (_cpty_idx[None, :] % 8), device=self.device)
    #     num_defs_per_batch = (self.batch_size+self.diffusion_engine.num_paths-1)//self.diffusion_engine.num_paths
    #     batch_size = min(self.batch_size, self.diffusion_engine.num_paths)
    #     while True:
    #         t = yield
    #         X = torch.as_tensor(self.diffusion_engine.X[t])
    #         t_prev_reset = self.prev_reset_arr[t]
    #         X_prev = torch.as_tensor(self.diffusion_engine.X[t_prev_reset])
    #         def_indicators = torch.as_tensor(self.diffusion_engine.def_indicators[t])
    #         def __gen_features(mean=None, std=None):
    #             nonlocal features_gpu
    #             for i in range((self.diffusion_engine.num_paths+batch_size-1)//batch_size):
    #                 features_gpu[:batch_size, :2*self.diffusion_engine.num_rates-1].copy_(X[:2*self.diffusion_engine.num_rates-1, i*batch_size:(i+1)*batch_size].T)
    #                 features_gpu[:batch_size, 2*self.diffusion_engine.num_rates-1:2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2].copy_(X[2*self.diffusion_engine.num_rates:2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-1, i*batch_size:(i+1)*batch_size].T)
    #                 features_gpu[:batch_size, 2*self.diffusion_engine.num_rates-1:2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2].relu_()
    #                 if t_prev_reset > 0:
    #                     features_gpu[:batch_size, 2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2].copy_(X_prev[:self.diffusion_engine.num_rates, i*batch_size:(i+1)*batch_size].T)
    #                 else:
    #                     features_gpu[:batch_size, 2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2].zero_()
    #                 if mean is not None:
    #                     features_gpu[:batch_size, :2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2] -= mean[None, :2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2]
    #                 if std is not None:
    #                     features_gpu[:batch_size, :2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2] /= (std[None, :2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2] + 1e-16)
    #                 for j in range(1, num_defs_per_batch):
    #                     features_gpu[j*batch_size:(j+1)*batch_size].copy_(features_gpu[:batch_size])
    #                 for j in range((self.diffusion_engine.num_defs_per_path+num_defs_per_batch-1)//num_defs_per_batch):
    #                     def_indicators_gpu.copy_(def_indicators[:, j*num_defs_per_batch: (j+1)*num_defs_per_batch, i*batch_size:(i+1)*batch_size].view(def_indicators.shape[0], -1).T)
    #                     features_gpu[:, 3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:].copy_((def_indicators_gpu[:, _cpty_idx//8] & _cpty_mask) != 0)
    #                     if mean is not None:
    #                         features_gpu[:, 3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:] -= mean[None, 3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:]
    #                     if std is not None:
    #                         features_gpu[:, 3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:] /= (std[None, 3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:] + 1e-16)
    #                     yield features_gpu
    #         yield __gen_features
    
    def _features_generator(self, load_from_device=False):
        assert isinstance(self.batch_size, int) and (self.batch_size >= 1) and (not (self.batch_size & (self.batch_size-1)))
        num_cpty = self.diffusion_engine.num_spreads-1
        features_gpu = torch.empty(self.batch_size, self.num_features, dtype=torch.float32, device=self.device)
        def_indicators_gpu = torch.empty(self.batch_size, (num_cpty+7)//8, dtype=torch.uint8, device=self.device)
        _cpty_idx = np.arange(num_cpty, dtype=np.int32)
        _cpty_mask = torch.tensor(1 << (_cpty_idx[None, :] % 8), device=self.device)
        num_defs_per_batch = (self.batch_size+self.diffusion_engine.num_paths-1)//self.diffusion_engine.num_paths
        batch_size = min(self.batch_size, self.diffusion_engine.num_paths)
        while True:
            t = yield
            t_prev_reset = self.prev_reset_arr[t]
            if t_prev_reset == 0:
                t_prev_reset = t
            if load_from_device:
                X = torch.as_tensor(self.diffusion_engine.d_X[self.diffusion_engine.max_coarse_per_reset], device=self.device)
                # very messy
                # TODO: clean this up
                shift = (t-1) % self.diffusion_engine.max_coarse_per_reset + 1
                X_prev = torch.as_tensor(self.diffusion_engine.d_X[self.diffusion_engine.max_coarse_per_reset-shift], device=self.device)
                def_indicators = torch.as_tensor(self.diffusion_engine.d_def_indicators[1])
            else:
                X = torch.as_tensor(self.diffusion_engine.X[t])
                X_prev = torch.as_tensor(self.diffusion_engine.X[t_prev_reset])
                def_indicators = torch.as_tensor(self.diffusion_engine.def_indicators[t])
            def __gen_features(mean=None, std=None):
                nonlocal features_gpu
                for i in range((self.diffusion_engine.num_paths+batch_size-1)//batch_size):
                    features_gpu[:batch_size, :2*self.diffusion_engine.num_rates-1].copy_(X[:2*self.diffusion_engine.num_rates-1, i*batch_size:(i+1)*batch_size].T)
                    features_gpu[:batch_size, 2*self.diffusion_engine.num_rates-1:2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2].copy_(X[2*self.diffusion_engine.num_rates:2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-1, i*batch_size:(i+1)*batch_size].T)
                    features_gpu[:batch_size, 2*self.diffusion_engine.num_rates-1:2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2].relu_()
                    if t > 0:
                        features_gpu[:batch_size, 2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2].copy_(X_prev[:self.diffusion_engine.num_rates, i*batch_size:(i+1)*batch_size].T)
                    else:
                        features_gpu[:batch_size, 2*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2].zero_()
                    if mean is not None:
                        features_gpu[:batch_size, :3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2] -= mean[None, :3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2]
                    if std is not None:
                        features_gpu[:batch_size, :3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2] /= (std[None, :3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2] + 1e-7)
                    for j in range(1, num_defs_per_batch):
                        features_gpu[j*batch_size:(j+1)*batch_size].copy_(features_gpu[:batch_size])
                    for j in range((self.diffusion_engine.num_defs_per_path+num_defs_per_batch-1)//num_defs_per_batch):
                        def_indicators_gpu.copy_(def_indicators[:, j*num_defs_per_batch: (j+1)*num_defs_per_batch, i*batch_size:(i+1)*batch_size].view(def_indicators.shape[0], -1).T)
                        features_gpu[:, 3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:].copy_((def_indicators_gpu[:, _cpty_idx//8] & _cpty_mask) != 0)
                        if mean is not None:
                            features_gpu[:, 3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:] -= mean[None, 3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:]
                        if std is not None:
                            features_gpu[:, 3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:] /= (std[None, 3*self.diffusion_engine.num_rates+self.diffusion_engine.num_spreads-2:] + 1e-7)
                        yield features_gpu
            yield __gen_features

    def _build_labels(self, as_cuda_tensor=False):
        if self.backward:
            return self._build_labels_backward(as_cuda_tensor)
        else:
            raise NotImplementedError
    
    def _build_labels_backward(self, as_cuda_tensor):
        t_spread_integral_now = torch.empty((self.diffusion_engine.num_spreads-1, self.diffusion_engine.num_paths), dtype=torch.float32, device=self.device)
        t_spread_integral_next = torch.empty((self.diffusion_engine.num_spreads-1, self.diffusion_engine.num_paths), dtype=torch.float32, device=self.device)
        t_mtm_next = torch.empty(self.diffusion_engine.d_mtm_by_cpty.shape[1:], dtype=torch.float32, device=self.device)
        t_rate_integral_now = torch.empty(self.diffusion_engine.d_dom_rate_integral.shape[1:], dtype=torch.float32, device=self.device)
        t_rate_integral_next = torch.empty(self.diffusion_engine.d_dom_rate_integral.shape[1:], dtype=torch.float32, device=self.device)
        t_def = torch.empty(self.diffusion_engine.d_def_indicators.shape[1:], dtype=torch.int8, device=self.device)
        t_labels_by_cpty = torch.empty(self.diffusion_engine.d_mtm_by_cpty.shape[1:], dtype=torch.float32, device=self.device)
        t_out = torch.empty((self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths), dtype=torch.float32, device=self.device)
        with cuda.devices.gpus[self.device.index]:
            d_spread_integral_now = cuda.as_cuda_array(t_spread_integral_now)
            d_spread_integral_next = cuda.as_cuda_array(t_spread_integral_next)
            d_mtm_next = cuda.as_cuda_array(t_mtm_next)
            d_rate_integral_now = cuda.as_cuda_array(t_rate_integral_now)
            d_rate_integral_next = cuda.as_cuda_array(t_rate_integral_next)
            d_def = cuda.as_cuda_array(t_def)
            d_labels_by_cpty = cuda.as_cuda_array(t_labels_by_cpty)
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
        d_spread_integral_next.copy_to_device(self.diffusion_engine.spread_integrals[self.diffusion_engine.num_coarse_steps, 1:])
        d_rate_integral_next.copy_to_device(self.diffusion_engine.dom_rate_integral[self.diffusion_engine.num_coarse_steps])
        accumulate = False
        for t in range(self.diffusion_engine.num_coarse_steps-1, -1, -1):
            d_spread_integral_now.copy_to_device(self.diffusion_engine.spread_integrals[t, 1:])
            d_rate_integral_now.copy_to_device(self.diffusion_engine.dom_rate_integral[t])
            d_mtm_next.copy_to_device(self.diffusion_engine.mtm_by_cpty[t+1])
            d_def.copy_to_device(self.diffusion_engine.def_indicators[t])
            self.__cuda_build_labels_backward(d_spread_integral_now, d_spread_integral_next, d_rate_integral_now, d_rate_integral_next, d_mtm_next, d_labels_by_cpty, t > 0, accumulate)
            self.__cuda_aggregate_survival(d_labels_by_cpty, d_def, d_out)
            if as_cuda_tensor:
                yield out.view(-1, 1)
            else:
                d_out.copy_to_host(out)
                yield out.reshape(-1, 1)
            if not accumulate:
                accumulate = True

    def _build_loss_backward(self, window):
        labels_gen_start = self._build_labels_backward(True)
        labels_gen_end = self._build_labels_backward(True)
        for t in range(self.diffusion_engine.num_coarse_steps, -1, -1):
            if t > self.diffusion_engine.num_coarse_steps-window:
                yield next(labels_gen_start).view(self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths)
            else:
                df = (torch.as_tensor(self.diffusion_engine.dom_rate_integral[t], dtype=torch.float32, device=self.device)-torch.as_tensor(self.diffusion_engine.dom_rate_integral[t+window], dtype=torch.float32, device=self.device)).exp_()
                yield next(labels_gen_start).view(self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths)-next(labels_gen_end).view(self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths)*df[None, :]
