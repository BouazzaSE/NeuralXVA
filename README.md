# NeuralXVA
NeuralXVA is a simulation code initially written during my PhD thesis for the paper **"Pathwise CVA Regressions With Oversimulated Defaults"** (published in [Mathematical Finance](https://doi.org/10.1111/mafi.12368), [preprint available on arXiv](https://arxiv.org/abs/2211.17005)) and then further used for **"CVA Sensitivities, Hedging and Risk"** (published on [Risk.net](https://www.risk.net/cutting-edge/7959752/cva-sensitivities-hedging-and-risk), [preprint available on arXiv](https://arxiv.org/abs/2407.18583)) and other current working papers. It implements:
* the generation of Monte Carlo paths of diffusion risk factors, default indicators and mark-to-markets in a multi-dimensional setup with multiple economies and counterparties;
* the learning of a path-wise CVA, FVA, dynamic IM, EC and KVA using one or more Neural Network regressions (L2 or quantile) to appropriately project the generated Monte Carlo samples of the payoffs.

We recommend users to first look at the `Demo - Simulations on the GPU.ipynb` notebook to get a handle of how the simulation data are stored and then the `Demo - Learning a path-wise CVA using hierarchical simulation.ipynb` notebook for a demo of the learning procedure.

## Fast simulations & trainings on the GPU

Since Monte-Carlo (nested or not) simulations are parallel in nature, they easily lend themselves to parallelization on GPUs. The simulation engine is implemented using custom CUDA routines for maximum speed on NVidia GPUs and implements the following optimizations (some of which are classic and in line with https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
* coalesced memory accesses for fast read/writes in the GPU global memory: this is achieved by having the Monte-Carlo scenario in the container array of any simulated process indexed using the last axis in the array and having a thread configuration such that two successive threads will correspond to two successive Monte-Carlo scenarios;
* all diffusion parameters are stored in the constant memory buffer: provides high-speed access to read-only data, which can be accessed by all threads at the same time using broadcasting;
* the shared memory is used as a fast buffer for storing the specifications of the products;
* using Python closures to construct special kernels for specific problem sizes: many loops depending on those sizes are then unrollable, which allows the compiler to put local arrays in registers when possible, allowing for faster accesses;
* default indicators are bit-packed in 8-bit integers: at any time-step, the default indicators vector will be of size `ceil(p/8)` where `p` is the number of counterparties, and the default indicator for the `i`-th counterparty will be stored in the `(i-1 mod 8)+1`-th bit of the `floor((i-1)/8)+1`-th component of that array (this can be read using bitwise operations), this roughly reduces the memory occupation of the default indicators by a factor of 8 compared to using booleans;
* batched simulations: we batch the simulations over the time axis, *i.e.* on the GPU we only simulate the trajectories over a time interval of size `cDtoH_freq` (see the parameters in the notebook) and copy each time the result back to the CPU where the different time batches are concatenated together to form the full trajectories from time `0` until maturity.

CUDA kernels are compiled just-in-time using Numba and this allows to test for various problem sizes without having to recompile any source code at the cost of a very small overhead when instanciating a `DiffusionEngine`. This approach is similar to runtime compilation using `nvrtc` (https://docs.nvidia.com/cuda/nvrtc/) in CUDA C/C++. As stated previously, it also allows to make some variables (*e.g.* problem sizes) known at compile-time in kernels defined inside Python closures, enabling many code optimizations when the Numba's JIT compiler goes over the kernels.

As opposed to most machine learning use-cases where the final product is the trained model and only inference is being performed when used by the end-user, in our case the training process itself is part of the final product because the distribution of the training data changes between two uses as the market data from which the diffusion parameters are inferred change. This calls for more care when writing the training procedures as they will be called at each use.

Thus, we chose to implement the learning schemes using PyTorch. This allows for interoperability with custom CUDA kernels or other CUDA-based packages implementing the CUDA Array Interface (*i.e.* arrays implement the `__cuda_array_interface__` attribute, see https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html). In particular, this allows us to reuse buffers from `DiffusionEngine` (which avoids unnecessary memory allocations) or to use fast cuSOLVER-based linear algebra routines from the `cupy` package on PyTorch tensors. PyTorch also exposes many of the CUDA internals, including whether or not to use pinned host memory (which we use by default to accelerate the data transfers between the CPU and the GPU).

## Citing
If you use this code in your work, we strongly encourage you to both cite this Github repository (with the corresponding identifier for the commit you are looking at) and, depending on the context of use, the papers describing the learning schemes used or more broadly my PhD manuscript:
```latex
@phdthesis{saadeddine2022learning,
  title={Learning From Simulated Data in Finance: XVAs, Risk Measures and Calibration},
  author={Saadeddine, Bouazza},
  year={2022},
  school={Universit{\'e} Paris-Saclay}
}
```

```latex
@article{abbas2023pathwise,
  title={Pathwise CVA regressions with oversimulated defaults},
  author={Abbas-Turki, Lokman A and Cr{\'e}pey, St{\'e}phane and Saadeddine, Bouazza},
  journal={Mathematical Finance},
  volume={33},
  number={2},
  pages={274--307},
  year={2023},
  publisher={Wiley Online Library}
}
```

```latex
@article{crepey2024cva,
  title={CVA Sensitivities, Hedging and Risk},
  author={Cr{\'e}pey, St{\'e}phane and Li, Botao and Nguyen, Hoang and Saadeddine, Bouazza},
  journal={Risk.net magazine July 2024, arXiv preprint arXiv:2407.18583},
  year={2024}
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
along with a smaller batch size, *i.e.* `(num_defs_per_path*num_paths)//64` instead of `(num_defs_per_path*num_paths)//32` for example:
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

Also make sure the GPU is enabled by navigating to *"Edit"* -> *"Notebook Settings"* and choosing *"GPU"* in *"Hardware Accelerator"*.
