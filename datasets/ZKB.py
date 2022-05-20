import functions.Fourier as Fourier
import functions.Chebyshev as Chebyshev
import functions.utils as utils
from jax import random
import jax.numpy as jnp

def ZKB(x, t, C, d, m):
  alpha = d / (d*(m - 1) + 2)
  beta = alpha / d
  k = alpha * (m - 1) / (2*m*d)
  u = utils.relu((C - k*x**2 / t**(2*beta))**(1/(m-1)))/ t**alpha
  return u

def porous_medium(n, m, N_samples, key=random.PRNGKey(1729), periodic=False, return_coefficients=True):
    features = []
    targets = []
    v = 0.5
    for i in range(N_samples):
        x = utils.grid(n, periodic=periodic)
        periodicity_t = False if return_coefficients else periodic
        t = (utils.grid(m, periodic=periodicity_t) + 1) / 2 + 0.05
        x, t = jnp.meshgrid(x, t, indexing="ij")
        keys = random.split(key, 3)
        key = keys[2]
        C = random.uniform(keys[0], ) * 0.09 + 0.01
        features.append(ZKB(x, t*0 + 0.05, C, 0.01, 2))
        targets.append(ZKB(x, t, C, 0.01, 2))
    features = jnp.stack(features)
    targets = jnp.stack(targets)
    if return_coefficients:
        if periodic:
            val_to_coeff = lambda x: jnp.transpose(utils.mixed_values_to_coefficients(jnp.transpose(x, axes=(1, 2, 0))), axes=(2, 0, 1))
        else:
            val_to_coeff = lambda x: jnp.transpose(utils.values_to_coefficients(jnp.transpose(x, axes=(1, 2, 0)), periodic=False), axes=(2, 0, 1))
    else:
        val_to_coeff = lambda x: x
    return val_to_coeff(jnp.stack(features)), val_to_coeff(jnp.stack(targets))
