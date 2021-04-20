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

from numba import cuda
import numpy as np
from simulation.diffusion_engine import DiffusionEngine
import time
import torch


class XVAEstimator:
    def __init__(self, diffusion_engine: DiffusionEngine, device: torch.device, num_hidden_layers, num_hidden_units, batch_size, \
        num_epochs, lr, holdout_size, *args, reset_weights=False, return_pred=True, linear=False, best_sol=True, **kwargs):
        self.diffusion_engine = diffusion_engine
        self.device = device
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.holdout_size = holdout_size
        self.reset_weights = reset_weights
        self.return_pred = return_pred
        self.linear = linear
        self.best_sol = best_sol

class XVAEstimatorPortfolio(XVAEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _train(self, features_gen, labels_as_cuda_tensors, measure_exec_times, ignore=None):
        exec_times = {'build_features': 0, 'build_labels': 0, 'save_state': 0, 'train': 0, 'num_trainings': 0}
        if ignore is None:
            ignore = tuple()
        self._estimator.reset()
        labels_gen = self._build_labels(labels_as_cuda_tensors)
        if self.backward:
            timesteps = range(self.diffusion_engine.num_coarse_steps, 0, -1)
        else:
            timesteps = range(1, self.diffusion_engine.num_coarse_steps+1)
            labels = next(labels_gen)
            if 0 not in ignore:
                self.saved_states[0] = (False, labels.mean())
        max_iter = self._estimator.total_iter
        if self.warmup:
            max_iter = (self._estimator.total_iter*self.diffusion_engine.num_coarse_steps+4)//5
        for t in timesteps:
            if measure_exec_times:
                exec_times['build_labels'] -= time.time()
            labels = next(labels_gen)
            if measure_exec_times:
                exec_times['build_labels'] += time.time()
            if t in ignore:
                continue
            labels_std = labels.std()
            if measure_exec_times:
                exec_times['build_features'] -= time.time()
            next(features_gen)
            features = features_gen.send(t)
            if measure_exec_times:
                exec_times['build_features'] += time.time()
            if labels_as_cuda_tensors:
                labels_std = labels_std.item()
            if labels_std > 1e-8:
                if measure_exec_times:
                    exec_times['num_trainings'] += 1
                    exec_times['train'] -= time.time()
                for i in range((max_iter+self._estimator.total_iter-1)//self._estimator.total_iter):
                    self._estimator.train(features, labels, max_iter=max_iter-i*self._estimator.total_iter)
                if measure_exec_times:
                    exec_times['train'] += time.time()
                if measure_exec_times:
                    exec_times['save_state'] -= time.time()
                self.saved_states[t] = (True, self._estimator.get_state())
                if measure_exec_times:
                    exec_times['save_state'] += time.time()
                if self.reset_weights:
                    self._estimator.reset()
                if self.compute_loss_surface:
                    self._loss_surface[t] = self._estimator.loss_hist
            else:
                labels_mean = labels.mean()
                if labels_as_cuda_tensors:
                    labels_mean = labels_mean.item()
                self.saved_states[t] = (False, labels_mean)
                if self.compute_loss_surface:
                    self._loss_surface[t] = np.nan
            if self.warmup:
                max_iter = (self._estimator.total_iter*4)//5
        if self.backward:
            labels = next(labels_gen)
            if 0 not in ignore:
                labels_mean = labels.mean()
                if labels_as_cuda_tensors:
                    labels_mean = labels_mean.item()
                self.saved_states[0] = (False, labels_mean)
        if self.compute_loss_surface:
            self._loss_surface[0] = 1
        return exec_times
    
    def train(self, features_gen=None, labels_as_cuda_tensors=False, measure_exec_times=False):
        if features_gen is None:
            features_gen = self._build_features()
        exec_times = self._train(features_gen, labels_as_cuda_tensors, measure_exec_times)
        return exec_times
    
    def _predict(self, t, features_gen, out, features=None):
        v, vv = self.saved_states[t]
        if v:
            if features is None:
                next(features_gen)
                features = features_gen.send(t)
            self._estimator.set_state(*vv)
            self._estimator.predict(features, out)
        else:
            out[:] = vv
    
    def predict(self, features_gen=None, as_cuda_array=False, flatten=True):
        if features_gen is None:
            features_gen = self._build_features()
        predicted_cva = torch.empty((self.diffusion_engine.num_defs_per_path*self.diffusion_engine.num_paths, 1), dtype=torch.float32, device=self.device)
        with cuda.devices.gpus[self.device.index]:
            d_predicted_cva = cuda.as_cuda_array(predicted_cva.view(self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths))
        if as_cuda_array:
            out = d_predicted_cva
        else:
            out = cuda.pinned_array((self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths), dtype=np.float32)
        if flatten:
            out = out.reshape(-1)
        while True:
            t = yield
            self._predict(t, features_gen, predicted_cva)
            if not as_cuda_array:
                d_predicted_cva.copy_to_host(out)
            yield out
