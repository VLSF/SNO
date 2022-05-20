# operations for periodic and non-periodic functions
import functions.Fourier as Fourier
import functions.Chebyshev as Chebyshev
import jax.numpy as jnp
import optax

from jax.lax import dynamic_slice
from functools import partial
from jax import jit, grad, vmap
from jax.lax import sign

def grid(n, periodic=False):
    # uniform or Chebyshev grid
    if periodic:
        return Fourier.Uniform_grid(n)
    else:
        return Chebyshev.Chebyshev_grid(n)

@partial(jit, static_argnums=[1, 2])
def integrate(coeff, axis, periodic=False):
    # compute indefinite integral from Chebyshev or Fourier coefficients
    if periodic:
        return Fourier.transform_coefficients(coeff, axis, "int")
    else:
        return Chebyshev.integrate(coeff, axis)

@partial(jit, static_argnums=[1, 2])
def differentiate(coeff, axis, periodic=False):
    # compute derivative from Chebyshev or Fourier coefficients
    if periodic:
        return Fourier.transform_coefficients(coeff, axis, "diff")
    else:
        return Chebyshev.differentiate(coeff, axis)

@partial(jit, static_argnums=1)
def values_to_coefficients(coeff, periodic=False):
    # obtain coefficients of trigonometric or Chebyshev series from values on uniform or Chebyshev grid
    if periodic:
        return Fourier.values_to_coefficients(coeff)
    else:
        return Chebyshev.values_to_coefficients(coeff)

def align_shapes(c1, c2, axes, mode="min"):
    # Produce aligned shapes for specified `axes`.
    # If mode == "min" the minimal length is kept, otherwise the maximal one in kept.
    # Example: For the input c1.shape = (4, 17, 3, 2), c2.shape = (22, 3, 5, 1), axes = ((0, 3, 1), (1, 2, 3)), mode = "max"
    # the output are tuples (4, 17, 3, 5), (22, 4, 5, 17)
    shape1, shape2 = c1.shape, c2.shape
    ax1, ax2 = axes
    compare = min if mode == "min" else max
    common_shape = [compare([shape1[i], shape2[j]]) for i, j in zip(ax1, ax2)]
    shapes_dict_1 = {i: shape for i, shape in zip(ax1, common_shape)}
    shapes_dict_2 = {i: shape for i, shape in zip(ax2, common_shape)}
    shape1_ = tuple([shape if i not in shapes_dict_1 else shapes_dict_1[i] for i, shape in enumerate(shape1)])
    shape2_ = tuple([shape if i not in shapes_dict_2 else shapes_dict_2[i] for i, shape in enumerate(shape2)])
    if mode=="min":
        return shape1_, shape2_
    else:
        return tuple([(0, sh1 - sh2) for sh1, sh2 in zip(shape1_, shape1)]), tuple([(0, sh1 - sh2) for sh1, sh2 in zip(shape2_, shape2)])

@partial(jit, static_argnums=1)
def expand(c, shape):
    # add zeros according to providen shape
    return jnp.pad(c, shape)

@partial(jit, static_argnums=1)
def truncate(c, shape):
    # truncate tensor according to providen shape
    return dynamic_slice(c, [0]*len(shape), shape)

@jit
def softplus(x, alpha=1.0, cut_off=35):
    a = (alpha*x < cut_off)
    return a*jnp.log(1 + jnp.exp(alpha*x*a))/alpha + jnp.logical_not(a) * x

@jit
def relu(x):
    return x*(x>0)

@jit
def complex_split_relu(x):
    return jnp.real(x)*(jnp.real(x)>0) + 1j*jnp.imag(x)*(jnp.imag(x)>0)

@jit
def complex_split_softplus(x, alpha=1.0):
    return softplus(jnp.real(x), alpha=alpha) + 1j*softplus(jnp.imag(x), alpha=alpha)

@jit
def cardioid(x):
    return (1 + jnp.cos(sign(x)))*x/2

@jit
def relu_I_quadrant(x):
    return x * (jnp.real(x) > 0) * (jnp.imag(x) > 0)

@jit
def complex_tanh(x):
    return jnp.tanh(abs(x)) * jnp.exp(1j*sign(x))

def activation_v(coeff, sigma, l=10, periodic=False):
    # perform refinement, transform coefficients to x-space, apply activation, find coefficients
    coeff = jnp.pad(coeff, [(0, l) for n in coeff.shape[:-1]] + [(0, 0)])
    if periodic:
        D = len(coeff.shape) - 1
        coeff = sigma(Fourier.coefficients_to_values(coeff, [n if i != D - 1 else 2*n-1 for i, n in enumerate(coeff.shape)]))
        return Fourier.values_to_coefficients(coeff)
    else:
        coeff = sigma(Chebyshev.coefficients_to_values(coeff))
        return Chebyshev.values_to_coefficients(coeff)

def get_differentiation_matrix(n):
    grid_points = grid(n)
    ones, i = jnp.ones(n, dtype=int), jnp.array(range(n), dtype=int)
    xi_minus_xj = jnp.outer(grid_points, ones) - jnp.outer(ones, grid_points)
    c = ones.at[0].set(2).at[-1].set(2)
    signs = (-1)**(jnp.outer(i, ones) + jnp.outer(ones, i))
    D_off_diagonal = jnp.outer(c, ones)*signs/(xi_minus_xj*jnp.outer(ones, c))
    mask = jnp.logical_not(jnp.isinf(D_off_diagonal))
    D_off_diagonal = D_off_diagonal*mask
    D_diagonal = jnp.sum(D_off_diagonal, axis=1)
    D = D_off_diagonal - jnp.diag(D_diagonal)
    return D

def get_interpolation_matrix(evaluate_at, m):
    known_at = grid(m)
    points = jnp.arange(m, dtype='int64')[::-1]
    weights = jnp.array(((-1)**points)*jnp.ones(m).at[0].set(1/2).at[-1].set(1/2), dtype='float64')
    n = len(evaluate_at)
    W = jnp.dot(jnp.ones((n, 1)), weights.reshape(1, m))/(jnp.dot(evaluate_at.reshape(-1, 1), jnp.ones((1, m))) - jnp.dot(jnp.ones((n, 1)), known_at.reshape(1, m)))
    mask = jnp.isinf(W)
    marked_rows = jnp.logical_not(jnp.sum(mask, axis=1)).reshape((-1, 1))
    W = jnp.nan_to_num(W*marked_rows, nan=1.0)
    return W

def get_interpolation_matrix_F(evaluate_at, m, is_real=False):
    k = Fourier.get_frequencies(m, is_real=is_real)
    if is_real:
        k = jnp.hstack([k, jnp.conj(k[1:])[::-1]])
    W = jnp.exp(jnp.outer(evaluate_at, k))
    return W

@partial(jit, static_argnums=[3, 5])
def update_params(params, x, y, optimizer, opt_state, loss):
  grads = grad(loss)(params, x, y)
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return params, opt_state

def mixed_values_to_coefficients(input):
    # (x, y, batch) -> (Fourier, Chebyshev, batch)
    input = vmap(Fourier.values_to_coefficients, in_axes=2, out_axes=2)(input)
    cheb_map = vmap(Chebyshev.values_to_coefficients)
    return cheb_map(jnp.real(input)) + 1j * cheb_map(jnp.imag(input))

def mixed_coefficients_to_values(input, n_fourier):
    # (Fourier, Chebyshev, batch) -> (x, y, batch)
    input = vmap(lambda x: Fourier.coefficients_to_values(x, (n_fourier, 1)), in_axes=2, out_axes=2)(input)
    return vmap(Chebyshev.coefficients_to_values)(input)
