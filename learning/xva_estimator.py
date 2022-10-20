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

from learning.misc import batch_mean, batch_std
from numba import cuda
import numpy as np
from simulation.diffusion_engine import DiffusionEngine
import time
import torch


class XVAEstimator:
    def __init__(self, diffusion_engine: DiffusionEngine, device: torch.device, num_hidden_layers, num_hidden_units, batch_size, \
        num_epochs, lr, holdout_size, *args, reset_weights=False, return_pred=True, linear=False, best_sol=True, refine_last_layer=True, **kwargs):
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
        self.refine_last_layer = refine_last_layer

class XVAEstimatorPortfolio(XVAEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__verbose__ = False
    
    def _train(self, batch_gen, exec_times, ignore=None, reset_estimator=True):
        if exec_times is not None:
            exec_times['save_state'] = 0
            exec_times['train'] = 0
            exec_times['num_trainings'] = 0
        if ignore is None:
            ignore = tuple()
        if reset_estimator:
            self._estimator.reset()
        for t, features_gen, labels_gen in batch_gen:
            if t in ignore:
                continue
            if self.__verbose__:
                print(f'* TRAINING {type(self).__name__} AT t={t}')
            if t == 0:
                if self._estimator.regr_type in ('mean', 'positive_mean'):
                    self.saved_states[0] = (False, batch_mean(labels_gen()).item())
                elif self._estimator.regr_type == 'quantile':
                    self.saved_states[0] = (False, torch.quantile(torch.cat(list(labels_gen()), dim=0), 1-self.quantile_level).item())
                else:
                    raise NotImplementedError
            else:
                if batch_std(labels_gen()).item() > 1e-7:
                    if exec_times is not None:
                        exec_times['num_trainings'] += 1
                        exec_times['train'] -= time.time()
                    self._estimator.train(features_gen, labels_gen)
                    if exec_times is not None:
                        exec_times['train'] += time.time()
                    if exec_times is not None:
                        exec_times['save_state'] -= time.time()
                    self.saved_states[t] = (True, self._estimator.get_state())
                    if exec_times is not None:
                        exec_times['save_state'] += time.time()
                    if self.reset_weights:
                        self._estimator.reset()
                    if self.compute_loss_surface:
                        self._loss_surface[t] = self._estimator.loss_hist
                else:
                    self.saved_states[t] = (False, batch_mean(labels_gen()).item())
                    if self.compute_loss_surface:
                        self._loss_surface[t] = np.nan
            yield
        if self.compute_loss_surface:
            self._loss_surface[0] = 1
    
    def train(self, batch_gen=None, labels_as_cuda_tensors=False, measure_exec_times=False, reset_estimator=True):
        if batch_gen is None:
            batch_gen = self._batch_generator(labels_as_cuda_tensors=labels_as_cuda_tensors, train_mode=True)
        exec_times = dict() if measure_exec_times else None
        for _ in self._train(batch_gen, exec_times, reset_estimator=reset_estimator):
            pass
        return exec_times

    def _post_predict(self, t, out):
        return out
    
    def _predict(self, t, features_gen, out):
        v, vv = self.saved_states[t]
        if v:
            next(features_gen)
            self._estimator.set_state(*vv)
            self._estimator.predict(features_gen.send(t), out)
        else:
            out[:] = vv
    
    def predict(self, features_gen=None, as_cuda_array=False, flatten=True, load_from_device=False):
        if features_gen is None:
            features_gen = self._features_generator(load_from_device=load_from_device)
        predicted_xva = torch.empty((self.diffusion_engine.num_defs_per_path*self.diffusion_engine.num_paths, 1), dtype=torch.float32, device=self.device)
        with cuda.devices.gpus[self.device.index]:
            d_predicted_xva = cuda.as_cuda_array(predicted_xva.view(self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths))
        if as_cuda_array:
            out = d_predicted_xva
        else:
            out = cuda.pinned_array((self.diffusion_engine.num_defs_per_path, self.diffusion_engine.num_paths), dtype=np.float32)
        if flatten:
            out = out.reshape(-1)
        while True:
            t = yield
            self._predict(t, features_gen, predicted_xva)
            out = self._post_predict(t, out)
            if not as_cuda_array:
                d_predicted_xva.copy_to_host(out)
            yield out
