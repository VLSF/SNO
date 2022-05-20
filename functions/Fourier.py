# periodic functions, domain [-1, 1)

import jax.numpy as jnp
from jax import jit
from functools import partial, reduce
from jax.config import config
config.update("jax_enable_x64", True)

def Uniform_grid(n):
    # uniform grid on [-1, 1)
    return jnp.array(jnp.linspace(-1, 1, n + 1)[:-1], dtype="float64")

def get_frequencies(n, is_real=False):
    # return frequencies used in derivative and indefinite integral
    # `is_real` refers to the last axis in FFT for real signals
    if is_real:
        k = 2*jnp.pi*1j*jnp.fft.rfftfreq(2*(n-1), d=1/(n-1))
    else:
        k = 2*jnp.pi*1j*jnp.fft.fftfreq(n, d=2/n)
    return k

def align_frequencies(values, forward=True):
    # input array has shape `(n1, n2, ..., c)`, where c stands for channels
    # if `forward` is `True` reorder elements according to the increase in magnitude of frequency
    # if `forward` is `False` perform inverse transposition to the case when forward is `True`
    D = len(values.shape) - 1
    if D == 1:
        return values
    else:
        transposition = [i for i in range(D-2)] + [i for i in range(D-1, D+1)]
        transposition = [D-2,] + transposition if D > 1 else transposition
    for i in range(D-1):
        order = jnp.argsort(abs(jnp.fft.fftfreq(len(values))))
        if not forward:
            order = jnp.array([x[0] for x in sorted(enumerate(order), key=lambda x: x[1])], dtype="int64")
        values = jnp.transpose(values[order], axes=transposition)
    return values

@jit
def reweight(values):
    # fix signs to comply with Fourier series
    D = len(values.shape)
    for i, s in enumerate(values.shape[:-1]):
        weight = (-1)**jnp.arange(s, dtype=int)
        values = values*weight.reshape([1 if j != i else s for j in range(D)])
    return values

@jit
def values_to_coefficients(values):
    # input array has shape `(n1, n2, ..., c)`, where c stands for channels
    # transform values of the function on the uniform grid to coefficients of trigonometric series
    # coefficients are reordered such that larger index corresponds to larger frequency
    coeff = jnp.fft.rfftn(values, axes=[i for i in range(len(values.shape)-1)], norm="forward")
    coeff = reweight(coeff)
    coeff = align_frequencies(coeff)
    return coeff

def coefficients_to_values(coefficients, shape):
    # x = coefficients_to_values(values_to_coefficients(x), x.shape) up to round-off error
    coeff = align_frequencies(coefficients, forward=False)
    coeff = reweight(coeff)
    values = jnp.fft.irfftn(coeff, s=shape[:-1], axes=[i for i in range(len(coefficients.shape)-1)], norm="forward")
    return values

@partial(jit, static_argnums=[1, 2, 3])
def transform_coefficients(coeff, axis, type, alpha=1.0):
    # input array has shape `(n1, n2, ..., c)`, where c stands for channels
    # `type` can be `diff`, `int`, `shift`
    # perform differentiation, integration or shift along `axis`
    D = len(coeff.shape) - 1
    is_real = True if axis == D - 1 else False
    k = get_frequencies(coeff.shape[axis], is_real=is_real)
    k = k.reshape([1 if i != axis else c for i, c in enumerate(coeff.shape)])
    k = align_frequencies(k)
    if type == "diff":
        return coeff*k
    elif type == "int":
        return coeff/k.at[tuple([0]*(D+1))].set(1.0)
    elif type == "shift":
        return jnp.exp(alpha*k)*coeff

def shift(coeff, axis, alpha):
    # input array has shape `(n1, n2, ..., nD)`
    # perform shift on `alpha` along given `axis`
    D = len(coeff.shape)
    is_real = True if axis == D else False
    k = get_frequencies(coeff.shape[axis], is_real=is_real)
    k = k.reshape([1 if i != axis else c for i, c in enumerate(coeff.shape)])
    k = align_frequencies(k)
    return jnp.exp(alpha*k)*coeff
