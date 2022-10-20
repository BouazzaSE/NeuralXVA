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

from collections import defaultdict, OrderedDict
from copy import deepcopy
import cupy as cp
from functools import partial
from itertools import chain
from learning.misc import batch_mean, batch_std
import math
import numpy as np
import torch
from typing import Optional, Tuple


def batch_iterate(features, labels, dest_features, dest_labels, batch_size):
    # handy way I came up with to reduce the number of unnecessary memory allocations
    # in particular this copies to the GPU (and ensures contiguity) only once every batch iteration
    assert features.shape[0] == labels.shape[0], 'features and labels must have same size along first axis'
    for batch_idx in range((features.shape[0]+batch_size-1)//batch_size):
        start_idx = batch_idx*batch_size
        tmp_features_batch = features[start_idx:(batch_idx+1)*batch_size]
        tmp_labels_batch = labels[start_idx:(batch_idx+1)*batch_size]
        eff_batch_size = tmp_features_batch.shape[0]
        dest_features[:eff_batch_size] = tmp_features_batch
        dest_labels[:eff_batch_size] = tmp_labels_batch
        yield start_idx, eff_batch_size, dest_features[:eff_batch_size], dest_labels[:eff_batch_size]

def weighted_batch_iterate(features, labels, weights, dest_features, dest_labels, dest_weights, batch_size):
    assert features.shape[0] == labels.shape[0], 'features and labels must have same size along first axis'
    if weights is not None:
        assert features.shape[0] == weights.shape[0], 'features and weights must have same size along first axis'
    for batch_idx in range((features.shape[0]+batch_size-1)//batch_size):
        start_idx = batch_idx*batch_size
        tmp_features_batch = features[start_idx:(batch_idx+1)*batch_size, :]
        eff_batch_size = tmp_features_batch.shape[0]
        dest_features[:eff_batch_size] = tmp_features_batch
        dest_features = dest_features[:eff_batch_size]
        dest_labels[:eff_batch_size] = labels[start_idx:(batch_idx+1)*batch_size]
        dest_labels = dest_labels[:eff_batch_size]
        if weights is not None:
            dest_weights[:eff_batch_size] = weights[start_idx:(batch_idx+1)*batch_size]
            dest_weights = dest_weights[:eff_batch_size]
        else:
            dest_weights = None
        yield start_idx, eff_batch_size, dest_features, dest_labels, dest_weights

def batch_iterate_features(features, dest_features, batch_size):
    for batch_idx in range((features.shape[0]+batch_size-1)//batch_size):
        start_idx = batch_idx*batch_size
        tmp_features_batch = features[start_idx:(batch_idx+1)*batch_size]
        eff_batch_size = tmp_features_batch.shape[0]
        dest_features[:eff_batch_size] = tmp_features_batch
        yield start_idx, eff_batch_size, dest_features[:eff_batch_size]

class GenericModelHiddenLayer(torch.jit.ScriptModule):
    def __init__(self, dim_in, dim_out):
        super(GenericModelHiddenLayer, self).__init__()
        self.W = torch.nn.Parameter(torch.empty(dim_in, dim_out, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.empty(1, dim_out, dtype=torch.float32))
        self.activation = torch.nn.Softplus()
        self.diff_activation = torch.nn.Sigmoid()

    def forward(self, x):
        z = torch.matmul(x, self.W) + self.b
        y = self.activation(z)
        return y
    
    @torch.jit.script_method
    def forward_backward(self, x, only_first_diff: bool):
        z = torch.matmul(x, self.W) + self.b
        diff_activation = self.diff_activation(z)
        if only_first_diff:
            W = self.W[None, 0]
        else:
            W = self.W
        y = self.activation(z)
        dy = W[None, :, :] * diff_activation[:, None, :]
        return y, dy

class GenericModelOutputLayer(torch.jit.ScriptModule):
    __constants__ = ['positive_mean']

    def __init__(self, dim_in, regr_type):
        super(GenericModelOutputLayer, self).__init__()
        self.positive_mean = False
        self.register_buffer('a', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('c', torch.tensor(0, dtype=torch.float32))

        if regr_type in ('mean', 'positive_mean', 'quantile', 'es'):
            self.W = torch.nn.Parameter(torch.empty(dim_in, 1, dtype=torch.float32))
            self.b = torch.nn.Parameter(torch.empty(1, 1, dtype=torch.float32))
        elif regr_type == 'quantile_es':
            self.W = torch.nn.Parameter(torch.empty(dim_in, 2, dtype=torch.float32))
            self.b = torch.nn.Parameter(torch.empty(1, 2, dtype=torch.float32))
        else:
            raise NotImplementedError
        if regr_type == 'positive_mean':
            self._activate_relu()
            self.positive_mean = True
    
    def _activate_relu(self):
        self.a.fill_(1)
    
    def _disable_relu(self):
        self.a.fill_(0)
    
    def forward(self, x):
        y = torch.matmul(x, self.W) + self.b
        if self.positive_mean:
            if self.a:
                return torch.relu(y) + self.c
        return y
    
    @torch.jit.script_method
    def forward_backward(self, x):
        # WARNING, THIS ASSUMES POSITIVE_MEAN = FALSE
        y = torch.matmul(x, self.W) + self.b
        dy = self.W[None, :, :]
        return y, dy

class AffineSoftplus(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(dim_in, dim_out, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.empty(1, dim_out, dtype=torch.float32))
        self.activation = torch.nn.Softplus()
        self.diff_activation = torch.nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(x, self.W) + self.b
        y = self.activation(z)
        return y
    
    @torch.jit.export
    def forward_diff(self, x: torch.Tensor, dy_prev: Optional[torch.Tensor], only_first_diff: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.matmul(x, self.W) + self.b
        diff_activation = self.diff_activation(z).unsqueeze(1)
        if only_first_diff:
            W = self.W[0].unsqueeze(0)
        else:
            W = self.W
        W = W.unsqueeze(0)
        y = self.activation(z)
        if dy_prev is not None:
            dy = (dy_prev @ W) * diff_activation
        else:
            dy = W * diff_activation
        return y, dy

class Affine(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(dim_in, dim_out, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.empty(1, dim_out, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.matmul(x, self.W) + self.b
        return y
    
    @torch.jit.export
    def forward_diff(self, x: torch.Tensor, dy_prev: Optional[torch.Tensor], only_first_diff: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.matmul(x, self.W) + self.b
        if only_first_diff:
            W = self.W[0].unsqueeze(0)
        else:
            W = self.W
        W = W.unsqueeze(0)
        if dy_prev is not None:
            dy = dy_prev @ W
        else:
            dy = W
        return y, dy
    
class ModelRandomAlphaPiecewiseAffine(torch.nn.Module):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, interpolation_nodes):
        super().__init__()
        h = []
        dim_in = input_dim-1
        for i in range(num_hidden_layers):
            h.append(AffineSoftplus(dim_in, num_hidden_units))
            dim_in = num_hidden_units
        self.h = torch.nn.ModuleList(h)
        self.o = Affine(num_hidden_units, interpolation_nodes.shape[0])
        self.register_buffer('x_mean', torch.zeros(1, input_dim, dtype=torch.float32))
        self.register_buffer('x_std', torch.ones(1, input_dim, dtype=torch.float32))
        self.register_buffer('y_mean', torch.zeros(1, 1, dtype=torch.float32))
        self.register_buffer('y_std', torch.ones(1, 1, dtype=torch.float32))
        self.register_buffer('interpolation_nodes', interpolation_nodes)
        self.register_buffer('interpolation_nodes_delta', interpolation_nodes[1:]-interpolation_nodes[:-1])
        self.init_weights()
    
    def init_weights(self):
        for l in chain(self.h, (self.o,)):
            torch.nn.init.normal_(l.W, mean=0., std=np.sqrt(1/l.W.shape[0]))
            torch.nn.init.zeros_(l.b)
    def forward(self, x):
        a = (x[:, 1:]-self.x_mean[:, 1:])/self.x_std[:, 1:]
        for l in self.h:
            a = l(a)
        a = self.o(a)
        a = a[:, 0, None]+((torch.minimum(x[:, 0, None], self.interpolation_nodes[None, 1:])-self.interpolation_nodes[None, :-1])*(self.interpolation_nodes[None, :-1]>=x[:, 0, None])*a[:, 1:]/self.interpolation_nodes_delta[None, :]).sum(1, keepdim=True)
        return a*self.y_std+self.y_mean

class GenericModel(torch.jit.ScriptModule):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, regr_type):
        super(GenericModel, self).__init__()

        h = []
        dim_in = input_dim
        for i in range(num_hidden_layers):
            h.append(GenericModelHiddenLayer(dim_in, num_hidden_units))
            dim_in = num_hidden_units
        self.h = torch.nn.ModuleList(h)

        self.o = GenericModelOutputLayer(num_hidden_units, regr_type)
        
        self.init_weights()

    def init_weights(self):
        for l in chain(self.h, (self.o,)):
            torch.nn.init.normal_(l.W, mean=0., std=np.sqrt(1/l.W.shape[0]))
            torch.nn.init.zeros_(l.b)

    @torch.jit.script_method
    def forward(self, x):
        a = x
        for l in self.h:
            a = l(a)
        return self.o(a)
    
    @torch.jit.script_method
    def forward_backward(self, x):
        # DON'T FORGET TO DIVIDE THE FINAL DIFFS BY STD_X & MULTIPLY WITH STD_Y !
        a, da = self.h[0].forward_backward(x, True)
        for l in self.h[1:]:
            a, l_da = l.forward_backward(a, False)
            da = da @ l_da
        a, l_da = self.o.forward_backward(a)
        da = da @ l_da
        return a, da

class GenericLinearModel(torch.jit.ScriptModule):
    def __init__(self, input_dim, regr_type):
        super(GenericLinearModel, self).__init__()
        if regr_type != 'mean':
            raise NotImplementedError
        self.W = torch.nn.Parameter(torch.empty(input_dim, 1, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.empty(1, 1, dtype=torch.float32))
        self.init_weights()

    def init_weights(self):
        torch.nn.init.normal_(self.W, mean=0., std=np.sqrt(1/self.W.shape[0]))
        torch.nn.init.zeros_(self.b)

    @torch.jit.script_method
    def forward(self, x):
        return torch.matmul(x, self.W) + self.b

@torch.jit.script
def _mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

@torch.jit.script
def _mse_loss_pointwise(y_pred, y_true):
    return (y_pred - y_true)**2

@torch.jit.script
def _weighted_mse_loss(y_pred, y_true, weights):
    return torch.sum((y_pred - y_true)**2*weights.view(-1, 1))/torch.sum(weights)

@torch.jit.script
def _safe_softplus(x):
    return torch.log(1+torch.exp(-torch.abs(x))) + torch.relu(x)

@torch.jit.script
def _vares_loss(alpha, y_pred, y_true):
    ind = (y_true < y_pred[0].view(-1, 1)).float()
    return torch.mean((ind - alpha) * y_pred[0].view(-1, 1) - ind * y_true \
        + torch.sigmoid(y_pred[1].view(-1, 1))*(y_pred[1].view(-1, 1)-y_pred[0].view(-1, 1)+(y_pred[0].view(-1, 1)-y_true)*ind/alpha) \
            - _safe_softplus(y_pred[1].view(-1, 1)))

@torch.jit.script
def _var_loss(alpha, y_pred, y_true):
    return torch.mean(torch.relu(y_true-y_pred)+alpha*y_pred)

@torch.jit.script
def _multiple_var_loss(alphas, y_pred, y_true):
    return torch.mean(torch.relu(y_true-y_pred)+alphas*y_pred)
@torch.jit.script
def _multiple_var_monotonicity_penalty(dy_pred):
    return torch.mean(torch.relu(dy_pred))

class GenericEstimator:
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, num_samples, batch_size, num_epochs, \
                lr, holdout_size, device, regr_type='mean', var_es_level=None, linear=False, best_sol=True, \
                refine_last_layer=True, multiple_var=False, interpolation_nodes=None, monotonicity_penalty=0.01):
        if not linear:
            if interpolation_nodes is not None:
                self.model = ModelRandomAlphaPiecewiseAffine(input_dim, num_hidden_layers, num_hidden_units, interpolation_nodes)
            else:
                self.model = GenericModel(input_dim, num_hidden_layers, num_hidden_units, regr_type)
        else:
            self.model = GenericLinearModel(input_dim, regr_type)
        self.linear = linear
        if device.type == 'cuda':
            self.model = self.model.cuda(device=device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.holdout_size = holdout_size
        self.device = device
        self.regr_type = regr_type
        if regr_type in ('mean', 'positive_mean'):
            self._loss_fct = _mse_loss
            self._loss_fct_pointwise = _mse_loss_pointwise
            self._weighted_loss_fct = _weighted_mse_loss
        elif regr_type in ('quantile', 'es'):
            self._loss_fct = partial(_var_loss, var_es_level)
        elif regr_type == 'quantile_es':
            self._loss_fct = partial(_vares_loss, var_es_level)
        else:
            raise NotImplementedError
        self.var_es_level = var_es_level
        self.best_sol = best_sol
        self.refine_last_layer = refine_last_layer
        self.multiple_var = multiple_var
        self.piecewise_affine_var = interpolation_nodes is not None
        self.monotonicity_penalty = monotonicity_penalty
        # TODO: move the following line to self.train and add a flag to specify whether we want that or not
        self.loss_hist = np.empty(num_epochs, dtype=np.float32)
        # INFO: no need to have num_samples and holdout_size in constructor
        # TODO: move the following to self.train and remove num_samples and holdout_size from the constructor
        self.total_iter = num_epochs*((num_samples-holdout_size+batch_size-1)//batch_size)
    
    def train(self, features_gen, labels_gen, max_iter=None, compute_heuristic=False, num_paths=None, valid_loss=False):
        # TODO: merge batch_mean & batch_std to avoid redundant copies
        self.t_features_mean = batch_mean(features_gen())
        self.t_features_std = batch_std(features_gen())
        self.t_labels_std = batch_std(labels_gen())
        self.t_labels_meanstd = batch_mean(labels_gen())/self.t_labels_std

        if compute_heuristic:
            raise NotImplementedError
            # TODO: fix the following and account for the new batch generator
            _t_features_holdout_reshaped = torch.reshape(t_features_holdout, (-1, num_paths, features.shape[1]))
            _t_labels_holdout_reshaped = torch.reshape(t_labels_holdout, (-1, num_paths, labels.shape[1]))
            t_features_h_1 = (_t_features_holdout_reshaped[1].cuda(self.device)-self.t_features_mean[None])/(self.t_features_std[None] + 1e-7)
            t_labels_h_1 = _t_labels_holdout_reshaped[1].cuda(self.device)/(self.t_labels_std[None] + 1e-7)
            t_features_h_2 = (_t_features_holdout_reshaped[2].cuda(self.device)-self.t_features_mean[None])/(self.t_features_std[None] + 1e-7)
            t_labels_h_2 = _t_labels_holdout_reshaped[2].cuda(self.device)/(self.t_labels_std[None] + 1e-7)

        best_loss = math.inf

        max_iter = max_iter if max_iter is not None else self.total_iter
        i = 0

        if not self.linear:
            if self.regr_type in ('mean', 'positive_mean', 'es'):
                h_aug = torch.empty((self.batch_size, self.model.o.W.shape[0]+1), dtype=torch.float32, device=self.device)
                h_aug[:, 0] = 1
                if self.regr_type == 'positive_mean':
                    self.model.o._disable_relu()
                    self.model.o.c.zero_()
        
        for e in range(self.num_epochs):
            if i==max_iter:
                break
            self.model.train()
            for features_batch, labels_batch in zip(features_gen(self.t_features_mean, self.t_features_std), labels_gen(None, self.t_labels_std)):
                if i==max_iter:
                    break
                if compute_heuristic:
                    raise NotImplementedError
                    # TODO: fix the following and account for the new batch generator
                    with torch.no_grad():
                        _f1 = self._loss_fct_pointwise(self.model(t_features_h_1), t_labels_h_1)
                        _f2 = self._loss_fct_pointwise(self.model(t_features_h_2), t_labels_h_2)
                        _m_f0_sq = (0.5*(_f1+_f2).mean().item())**2
                        _m_f0f0 = 0.5*(_f1**2+_f2**2).mean().item()
                        _var = _m_f0f0 - _m_f0_sq
                        _m_f1f2 = (_f1*_f2).mean().item()
                        try:
                            print('[i={}] "optimal" N = {}, _m_f0f0 = {}, _m_f1f2 = {}, _m_f0_sq = {}, _var = {}'.format(i, \
                                np.sqrt((abs(_m_f0f0-_m_f1f2)+1e-7)/(abs(_m_f0_sq-_m_f1f2)+1e-7)), _m_f0f0, _m_f1f2, _m_f0_sq, _var))
                        except:
                            print(_m_f0f0, _m_f1f2, _m_f0_sq, _m_f1f2)
                            raise
                if self.multiple_var and (not self.piecewise_affine_var):
                    y_pred, dy_pred = self.model.forward_backward(features_batch)
                else:
                    y_pred = self.model(features_batch)
                if not self.multiple_var:
                    loss = self._loss_fct(y_pred, labels_batch)
                else:
                    loss = _multiple_var_loss((features_batch[:, 0]*self.t_features_std[0]+self.t_features_mean[0])[:, None], y_pred, labels_batch)
                    if not self.piecewise_affine_var:
                        loss = loss + self.monotonicity_penalty*_multiple_var_monotonicity_penalty(dy_pred)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i += 1
                        
            with torch.no_grad():
                self.model.eval()
                total_loss = 0
                update_c = False
                if self.multiple_var:
                    total_monotonicity_penalty = 0

                if (not self.linear) and (self.regr_type in ('mean', 'positive_mean')):
                    if e == self.num_epochs//2:
                        if self.refine_last_layer:
                            j = 0
                            self.model.o.W.zero_()
                            self.model.o.b.zero_()
                            for features_batch, labels_batch in zip(features_gen(self.t_features_mean, self.t_features_std), labels_gen(None, self.t_labels_std)):
                                # features_batch -= self.t_features_mean[None]
                                # features_batch /= (self.t_features_std[None] + 1e-16)
                                # labels_batch /= (self.t_labels_std[None] + 1e-16)
                                h = features_batch
                                for l in self.model.h:
                                    h = l(h)
                                h_aug[:, 1:] = h
                                with cp.cuda.Device(self.device.index):
                                    sol, _, _, _ = cp.linalg.lstsq(cp.asarray(h_aug), cp.asarray(labels_batch), rcond=None)
                                    sol = torch.as_tensor(sol, device=self.device)
                                self.model.o.W.add_(sol[1:h_aug.shape[1]])
                                self.model.o.b.add_(sol[:1])
                                j += 1
                            self.model.o.W /= j
                            self.model.o.b /= j
                        if self.regr_type == 'positive_mean':
                            self.model.o._activate_relu()
                    
                    if self.regr_type == 'positive_mean':
                        update_c = self.model.o.a.item()
                        if update_c:
                            y_pred_mean = 0
                
                if (e < self.num_epochs//2) and (not self.linear) and (self.regr_type=='positive_mean'):
                    self.model.o._activate_relu()

                k = 1
                for features_batch, labels_batch in zip(features_gen(self.t_features_mean, self.t_features_std), labels_gen(None, self.t_labels_std)):
                    if self.multiple_var and (not self.piecewise_affine_var):
                        y_pred, dy_pred = self.model.forward_backward(features_batch)
                    else:
                        y_pred = self.model(features_batch)
                    total_loss += self._loss_fct(y_pred, labels_batch) #
                    if not self.multiple_var:
                        total_loss += self._loss_fct(y_pred, labels_batch)
                    else:
                        total_loss += _multiple_var_loss((features_batch[:, 0]*self.t_features_std[0]+self.t_features_mean[0])[:, None], y_pred, labels_batch)
                        if not self.piecewise_affine_var:
                            total_monotonicity_penalty += self.monotonicity_penalty*_multiple_var_monotonicity_penalty(dy_pred)
                    if update_c:
                        y_pred_mean += y_pred.sum(0) #
                    k += 1
                total_loss /= k
                if update_c:
                    y_pred_mean /= k
                    bias_adj = torch.relu_(self.model.o.c + (self.t_labels_meanstd - y_pred_mean).view(self.model.o.c.shape))-self.model.o.c
                    self.model.o.c += bias_adj
                total_loss = total_loss.item()
                if self.multiple_var and (not self.piecewise_affine_var):
                    total_monotonicity_penalty = total_monotonicity_penalty.item() / k
                    total_loss += total_monotonicity_penalty
                    # TODO: do the same for the holdout version
                if self.holdout_size > 0:
                    raise NotImplementedError
                    # TODO: fix the following and account for the new batch generator
                    total_validation_loss = 0
                    for _, eff_batch_size, features_batch, labels_batch, weights_batch in weighted_batch_iterate(t_features_holdout, t_labels_holdout, t_weights_holdout, t_features_batch, t_labels_batch, t_weights_batch, self.batch_size):
                        # features_batch -= self.t_features_mean[None]
                        # features_batch /= (self.t_features_std[None] + 1e-16)
                        # labels_batch /= (self.t_labels_std[None] + 1e-16)
                        if weights_batch is None:
                            total_validation_loss += self._loss_fct(self.model(features_batch), labels_batch)*eff_batch_size/t_features_holdout.shape[0]
                        else:
                            weights_batch_sum = weights_batch.sum()
                            total_validation_loss += self._weighted_loss_fct(self.model(features_batch), labels_batch, weights_batch)*weights_batch_sum/weights_holdout_sum
                    total_validation_loss = total_validation_loss.item()
                else:
                    total_validation_loss = np.nan
                
                if (e < self.num_epochs//2) and (not self.linear) and (self.regr_type=='mean'):
                    self.model.o._disable_relu()

            if valid_loss:
                self.loss_hist[e] = total_validation_loss
            else:
                self.loss_hist[e] = total_loss
            
            if self.best_sol:
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_model_state = deepcopy(self.model.state_dict())
                    best_optimizer_state = deepcopy(self.optimizer.state_dict())
        
        if self.best_sol:
            self.model.load_state_dict(best_model_state)
            self.optimizer.load_state_dict(best_optimizer_state)

        # if (not self.linear) and (self.regr_type=='es'):
        #     j = 0
        #     t_labels_std = batch_std(((y-self.model(x)).relu_().div_(self.alpha).add(self.model(x)) for x, y in zip(features_gen(self.t_features_mean, self.t_features_std), labels_gen(None, self.t_labels_std))))
        #     self.model.o.W.zero_()
        #     self.model.o.b.zero_()
        #     for features_batch, labels_batch in zip(features_gen(self.t_features_mean, self.t_features_std), labels_gen(None, self.t_labels_std)):
        #         h = features_batch
        #         for l in self.model.h:
        #             h = l(h)
        #         h_aug[:, 1:] = h
        #         with cp.cuda.Device(self.device.index):
        #             sol, _, _, _ = cp.linalg.lstsq(cp.asarray(h_aug), cp.asarray(labels_batch), rcond=None)
        #             sol = torch.as_tensor(sol, device=self.device)
        #         self.model.o.W.add_(sol[1:h_aug.shape[1]])
        #         self.model.o.b.add_(sol[:1])
        #         j += 1
        #     self.model.o.W /= j
        #     self.model.o.b /= j
    
    def predict(self, features_gen, out):
        assert out.shape[1]==self.t_labels_std.shape[0], 'wrong shape for given output array'
        assert out.dtype in (np.float32, torch.float32), 'wrong dtype for given output array'
        t_out = torch.as_tensor(out)
        with torch.no_grad():
            self.model.eval()
            for i, features_batch in enumerate(features_gen(self.t_features_mean, self.t_features_std)):
                t_out[i*self.batch_size:(i+1)*self.batch_size].copy_(self.model(features_batch)*(self.t_labels_std[None] + 1e-7))
        return out
    
    def get_state(self, to_host=False):
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        if to_host:
            # assuming that all tensors are on GPU, hence no need for deepcopy since .cpu() will create a new tensor anyway
            model_state = OrderedDict([(k, v.cpu()) for k, v in model_state.items()])
            optimizer_state = OrderedDict([(k, v.cpu()) for k, v in optimizer_state.items()])
        else:
            # performing deep copies since .state_dict() contains references
            model_state = deepcopy(model_state)
            optimizer_state = deepcopy(optimizer_state)
        return model_state, optimizer_state, self.t_features_mean.cpu().numpy(), self.t_features_std.cpu().numpy(), self.t_labels_std.cpu().numpy()
    
    def set_state(self, model_state, optimizer_state, features_mean, features_std, labels_std):
        self.t_features_mean = torch.tensor(features_mean, device=self.device)
        self.t_features_std = torch.tensor(features_std, device=self.device)
        self.t_labels_std = torch.tensor(labels_std, device=self.device)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
    
    def reset(self):
        self.model.init_weights()
        self.optimizer.state = defaultdict(dict)
