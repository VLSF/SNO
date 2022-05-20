# non-periodic functions, domain [-1, 1]

import jax.numpy as jnp
from jax import jit
from functools import partial, reduce
from jax.config import config
from jax.scipy.linalg import solve_triangular
from jax.lax import dot_general
config.update("jax_enable_x64", True)

def Chebyshev_grid(n):
    # Chebyshev grid (second kind)
    return jnp.array(jnp.cos(jnp.arange(n, dtype='int64')*jnp.pi/(n-1))[::-1], dtype='float64')

@jit
def values_to_coefficients(values):
    # input array has shape `(n1, n2, ..., c)`, where c stands for channels
    # transform values of the function on the Chebyshev grid to coefficients of Chebyshev series
    D = values.shape
    transposition = [i for i in range(1, len(D)-1)] + [0, len(D)-1]
    for N in D[:-1]:
        values = jnp.real(jnp.fft.rfft(jnp.pad(values, [(0, N-2)] + [(0, 0)]*(len(D)-1), mode="reflect"), axis=0))/(N-1)
        values = values.at[0].set(values[0]/2).at[-1].set(values[-1]/2)
        values = values*((-1)**jnp.arange(N, dtype=int)).reshape([-1]+[1]*(len(D)-1))
        values = jnp.transpose(values, transposition)
    return values

@jit
def coefficients_to_values(coefficients):
    # x = coefficients_to_values(values_to_coefficients(x)) up to round-off error
    D = coefficients.shape
    transposition = [i for i in range(1, len(D)-1)] + [0, len(D)-1]
    for N in D[:-1]:
        coefficients = coefficients*((-1)**jnp.arange(N, dtype=int)).reshape([-1]+[1]*(len(D)-1))
        coefficients = jnp.pad((N-1)*coefficients.at[0].set(coefficients[0]*2).at[-1].set(coefficients[-1]*2), [(0, N-2)] + [(0, 0)]*(len(D)-1), mode="reflect")
        coefficients = jnp.real(jnp.fft.rfft(coefficients, axis=0))/(N-1)/2
        coefficients = jnp.transpose(coefficients, transposition)
    return coefficients

@partial(jit, static_argnums=1)
def differentiate(coeff, axis):
    # find derivative along `axis`
    shape = coeff.shape
    n = shape[axis]
    A = -jnp.eye(n-1, k=+2) + jnp.eye(n-1, k=0)
    A = A.at[0, 0].set(2)
    transposition = [axis] + [i for i in range(len(shape)) if i != axis]
    inv_transposition = [i for i in range(1, axis+1)] + [0] + [i for i in range(axis+1, len(shape))]
    coeff = jnp.transpose(coeff, transposition)
    shape_ = coeff.shape
    w = ((jnp.arange(0, shape_[0]-1)+1)*2).reshape([-1] + [1])
    coeff = solve_triangular(A, coeff.reshape((shape_[0], -1))[1:]*w)
    coeff = coeff.reshape([shape_[0]-1] + list(shape_[1:]))
    coeff = jnp.transpose(coeff, inv_transposition)
    return coeff

@partial(jit, static_argnums=1)
def integrate(coeff, axis):
    # find indefinite integral along `axis`
    n = coeff.shape[axis]
    sh = [-1 if i == axis else 1 for i in range(len(coeff.shape))]
    w_0 = jnp.hstack([jnp.array([1, 1/4]), 1 / jnp.arange(3, n+2) / 2]).reshape(sh)
    w_1 = jnp.hstack([jnp.array([0, 1/4]), - 1 /  jnp.arange(1, n) / 2]).reshape(sh)
    coeff = jnp.pad(coeff, [(0, 1) if i == axis else (0, 0) for i in range(len(coeff.shape))])
    coeff = jnp.roll(w_0*coeff, 1, axis=axis) + jnp.roll(w_1*coeff, -1, axis=axis)
    # correct to have indefinite integral from -1 to x
    coeff = coeff.at[tuple([0]*(len(coeff.shape) - 1))].set(0)
    w = (-1)**jnp.arange(n+1, dtype=int)
    coeff = jnp.moveaxis(coeff, axis, 0)
    coeff = coeff.at[0].set(-dot_general(w + 0., coeff, (((0,), (0,)), ((), ()))))
    coeff = jnp.moveaxis(coeff, 0, axis)
    return coeff
