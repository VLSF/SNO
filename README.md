# SNO
Spectral Neural Operator is a neural network that performs mapping between two functions given as Chebyshev or Fourier series. More details about SNO are available in the article [V. Fanaskov, I. Oseledets, Spectral Neural Operators](https://arxiv.org/abs/2205.10573). In the repository you can find implementation of basic operations with polynomials, architectures we discuss in the article and scripts for dataset generation.

We cover main functionality with Jupyter notebooks:
+ functions
  + [Fourier](https://github.com/VLSF/SNO/blob/main/notebooks/functions/Fourier.ipynb) -- operations with Fourier series.
  + [Chebyshev](https://github.com/VLSF/SNO/blob/main/notebooks/functions/Chebyshev.ipynb) -- operations with Chebyshev series.
  + [utils](https://github.com/VLSF/SNO/blob/main/notebooks/functions/Utilities.ipynb) -- convenience routines, wrappers, activation functions, tools for reshaping, etc.
+ architectures
  + [FNO](https://github.com/VLSF/SNO/blob/main/notebooks/architectures/FNO.ipynb) -- Fourier Neural Operator.
  + [DeepONet](https://github.com/VLSF/SNO/blob/main/notebooks/architectures/DeepONet.ipynb) -- Deep Operator Network.
  + [SNO](https://github.com/VLSF/SNO/blob/main/notebooks/architectures/SNO.ipynb) -- Spectral Neural Operator in Chebyshev basis.
  + [fSNO](https://github.com/VLSF/SNO/blob/main/notebooks/architectures/fSNO.ipynb) -- Spectral Neural Operator in Fourier basis.
  + [SNOx](https://github.com/VLSF/SNO/blob/main/notebooks/architectures/SNOx.ipynb) -- Spectral Neural Operator on Chebyshev grid.
  + [fSNOx](https://github.com/VLSF/SNO/blob/main/notebooks/architectures/fSNOx.ipynb) -- Spectral Neural Operator on the uniform grid.
  + [SNOxw](https://github.com/VLSF/SNO/blob/main/notebooks/architectures/SNOxw.ipynb) -- composition of SNO and SNOx.
  + [fSNOxw](https://github.com/VLSF/SNO/blob/main/notebooks/architectures/fSNOxw.ipynb) -- composition of fSNO and fSNOx.
+ datasets
  + [Parametric ODE](https://github.com/VLSF/SNO/blob/main/notebooks/datasets/Parametric%20ODE.ipynb)
  + [Shift](https://github.com/VLSF/SNO/blob/main/notebooks/datasets/Square%20shift.ipynb)
  + [Indefinite integrals](https://github.com/VLSF/SNO/blob/main/notebooks/datasets/Indefinite%20integrals.ipynb)
  + [Differentiation](https://github.com/VLSF/SNO/blob/main/notebooks/datasets/Differentiation.ipynb)
  + [Elliptic](https://github.com/VLSF/SNO/blob/main/notebooks/datasets/Elliptic.ipynb)
  + [Burgers](https://github.com/VLSF/SNO/blob/main/notebooks/datasets/Burgers.ipynb)
  + [KdV](https://github.com/VLSF/SNO/blob/main/notebooks/datasets/KdV%20exact.ipynb)
  + [Breather](https://github.com/VLSF/SNO/blob/main/notebooks/datasets/Breather.ipynb)

Most datasets can be efficiently generated using provided scripts in a matter of minutes. Two notable exceptions are Burgers equation and elliptic equation in D=2. You can use [this link](https://drive.google.com/drive/folders/1x5-rPf5Rg-KYy7tB9ibxsnFosRMmjIrb?usp=sharing) to access complete datasets for these two cases. Content of the folder, and instructions how to process data, can be found in [this notebook](https://github.com/VLSF/SNO/blob/main/notebooks/datasets/Working%20with%20datasets.ipynb).
