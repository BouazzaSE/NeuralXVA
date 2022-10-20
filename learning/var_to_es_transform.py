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

from learning.ec_estimator_portfolio import ECEstimatorPortfolio
from learning.generic_estimator import GenericEstimator
from learning.xva_estimator import XVAEstimatorPortfolio
import numpy as np
import torch


class VaRtoESTransform(XVAEstimatorPortfolio):
    def __init__(self, ec_term: ECEstimatorPortfolio, prev_reset_arr, backward, warmup, compute_loss_surface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features = 3*self.diffusion_engine.num_rates+2*(self.diffusion_engine.num_spreads-1)-1
        self.ec_term = ec_term
        self.backward = backward
        self.warmup = warmup
        regr_type = 'mean'
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
        if not self.backward:
            raise NotImplementedError
        pass
    
    def _batch_generator(self, labels_as_cuda_tensors=True, load_from_device=False, train_mode=False):
        assert isinstance(self.batch_size, int) and (self.batch_size >= 1) and (not (self.batch_size & (self.batch_size-1)))
        features_gen = self._features_generator(load_from_device=load_from_device)
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

    def _build_labels(self, as_cuda_tensor=False, load_from_device=False, train_mode=False):
        if self.backward:
            return self._build_labels_backward(as_cuda_tensor, load_from_device=load_from_device)
        else:
            raise NotImplementedError
    
    def _build_labels_backward(self, as_cuda_tensor, load_from_device=False):
        if load_from_device:
            raise NotImplementedError
        ec_predictor = self.ec_term.predict(as_cuda_array=True, flatten=False, load_from_device=False)
        ec_labels_gen = self.ec_term._build_labels(as_cuda_tensor)
        for t in range(self.diffusion_engine.num_coarse_steps, -1, -1):
            next(ec_predictor)
            _ec = torch.as_tensor(ec_predictor.send(t), dtype=torch.float32, device=self.device)
            _ec_labels = next(ec_labels_gen).view(self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths)
            # _ec_labels.mul_(_ec.le_(_ec_labels)).div_(self.ec_term.quantile_level)
            _ec.add_((_ec_labels-_ec).relu_().div_(self.ec_term.quantile_level))
            if as_cuda_tensor:
                yield _ec.view(-1, 1)
            else:
                raise NotImplementedError

    def _build_loss_backward(self, window):
        raise NotImplementedError