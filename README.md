# NeuralXVA
NeuralXVA is a simulation code written for the paper **"Hierarchical Simulation for Learning With Defaults"** by Abbas-Turki Lokman, Crépey Stéphane and Saadeddine Bouazza (see citation below), implementing:
* the generation via Monte Carlo of paths of diffusion risk factors, default indicators and mark-to-markets in a multi-dimensional setup with multiple economies and counterparties;
* the learning of a CVA based on the generated Monte Carlo samples of the payoffs (codebase to be extended soon for other XVAs).

We recommend first users to first look at the `Demo - Simulations on the GPU.ipynb` notebook to get a handle of how the simulation data is stored and then the `Demo - Learning a CVA using hierarchical simulation.ipynb` notebook for a demo of the learning procedure.

The diffusion engine is implemented using custom CUDA routines for maximum speed on NVidia GPUs. CUDA kernels are compiled just-in-time using Numba, which allows to test for various problem sizes without having to recompile any source code at the cost of a very small overhead when instanciating a `DiffusionEngine`.

The learning of the CVA is done via Neural Network regression and uses PyTorch. This allows for interoperability with custom CUDA kernels or other CUDA-based packages implementing the CUDA Array Interface (*ie* arrays have the `__cuda_array_interface__` attribute). In particular, this allows us to reuse buffers from `DiffusionEngine` or to use fast cuSOLVER-based linear algebra routines from the `cupy` package on PyTorch tensors.

## Citing
If you use this code in your work, we strongly encourage you to both cite this Github repository (with the corresponding identifier for the commit you are looking at) and the papers describing our learning schemes:
```latex
@unpublished{deepxva,
  title={Hierarchical Simulation for Learning With Defaults},
  author={Abbas-Turki, Lokman A and Cr{\'e}pey, St{\'e}phane and Saadeddine, Bouazza},
  year={\ndd},
  note={unpublished}
}

@article{CrepeyHoskinsonSaadeddine2019,
  title={XVA analysis from the balance sheet},
  author={Albanese, Claudio and Cr{\'e}pey, St{\'e}phane and Hoskinson, Rodney and Saadeddine, Bouazza},
  journal={Quantitative Finance},
  pages={1--25},
  year={2020},
  publisher={Taylor \& Francis}
}
```

## Working versions of packages
The code has been tested with the following package versions:
* `cupy >= 8.5.0`
* `numba >= 0.51.2`;
* `numexpr >= 2.7.1`;
* `numpy >= 1.19.1`;
* `torch >= 1.7.0`;

and Python version `3.8.5`.

## Using on Google Colaboratory
One must first uncomment the first cell in the demo notebooks. They can then be run on Google Colaboratory, although one has to limit the problem size in order not to run into out-of-memory errors due to limitations on Google's end.

In particular, the following:
```python
num_paths = 2**14
num_inner_paths = 256
num_defs_per_path = 128
```
along with a smaller batch size, *ie* `(num_defs_per_path*num_paths)//64` instead of `(num_defs_per_path*num_paths)//32` for example:
```python
# learner with default indicators
cva_estimator_portfolio_def = CVAEstimatorPortfolioDef(prev_reset_arr, True, False, False, diffusion_engine, 
                                                       device, 1, 2*(num_rates+num_spreads), (num_defs_per_path*num_paths)//64, 
                                                       4, 0.01, 0, reset_weights=False, linear=False, best_sol=True)

# learner with default intensities
cva_estimator_portfolio_int = CVAEstimatorPortfolioInt(prev_reset_arr, True, False, False, diffusion_engine, 
                                                       device, 1, 2*(num_rates+num_spreads), (num_defs_per_path*num_paths)//64, 
                                                       4, 0.01, 0, reset_weights=False, linear=False, best_sol=True)

# notice we do 4 epochs instead of 8 this time in order to keep the same number of total SGD steps
```
works on Google Colaboratory. *(ignore the remark on the batch size if using the `Demo - Simulations on the GPU.ipynb` notebook)*
