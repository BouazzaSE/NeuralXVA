import numpy as np
import torch


def batch_mean(batch_gen):
    # assuming all batches are of the same size, should be the case when using powers of 2
    mean = 0
    for i, batch in enumerate(batch_gen):
        mean += batch.mean(0)
    mean /= (i+1)
    return mean

def batch_std(batch_gen):
    # assuming all batches are of the same size, should be the case when using powers of 2
    std = 0
    for i, batch in enumerate(batch_gen):
        std += batch.var(0)
    std /= (i+1)
    std.sqrt_()
    return std

def predict(estimator, num_coarse_steps, num_defs_per_path, num_paths, stop_at=0):
    features_gen = estimator._features_generator()
    predictor = estimator.predict(features_gen=features_gen, as_cuda_array=True, flatten=False)
    predicted_xva = np.empty((num_coarse_steps+1, num_defs_per_path*num_paths), dtype=np.float32)
    _v = predicted_xva.reshape(num_coarse_steps+1, num_defs_per_path, num_paths)
    for t in range(num_coarse_steps, stop_at-1, -1):
        next(predictor)
        _v[t] = predictor.send(t)
    return predicted_xva

def predict_only_stats(estimator, stats, num_coarse_steps, recompute_at_zero=False):
    for stat in stats:
        assert isinstance(stat, str) or isinstance(stat, tuple)
        assert stat=='mean' or stat=='std' or ((len(stat)==2) and stat[0]=='quantile')
    device = estimator.device
    predictor = estimator.predict(as_cuda_array=True, flatten=False)
    predicted_xva_stats = {stat: np.empty(num_coarse_steps+1, dtype=np.float32) for stat in stats}
    end = 0 if recompute_at_zero else -1
    for t in range(num_coarse_steps, end, -1):
        next(predictor)
        _v = torch.as_tensor(predictor.send(t), dtype=torch.float32, device=device).view(-1)
        for stat, predicted_xva_stat in predicted_xva_stats.items():
            if stat == 'mean':
                _stat = _v.mean().item()
            elif stat == 'std':
                _stat = _v.std().item()
            else:
                _stat = _v.quantile(stat[1]).item()
            predicted_xva_stat[t] = _stat
    if recompute_at_zero:
        batch_gen = estimator._batch_generator(labels_as_cuda_tensors=True, train_mode=False)
        for t, _, labels_gen in batch_gen:
            if t == 0:
                if estimator._estimator.regr_type in ('mean', 'positive_mean'):
                    estimator.saved_states[0] = (False, batch_mean(labels_gen()).item())
                elif estimator._estimator.regr_type == 'quantile':
                    estimator.saved_states[0] = (False, torch.quantile(torch.cat(list(labels_gen()), dim=0), 1-estimator.quantile_level).item())
                else:
                    raise NotImplementedError
                for stat, predicted_xva_stat in predicted_xva_stats.items():
                    predicted_xva_stat[0] = 0 if stat=='std' else estimator.saved_states[0][1]
                break
    return predicted_xva_stats