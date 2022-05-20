import functions.Fourier as Fourier
import functions.Chebyshev as Chebyshev
import functions.utils as utils
from jax import random
import jax.numpy as jnp

def Fourier_series(ord, n, N_samples, key=random.PRNGKey(1729), shift=0, sigma=1.0, mu=0.0, periodic=False, return_coefficients=True, normal=True):
    features = []
    targets = []
    x = utils.grid(n, periodic=periodic)
    for i in range(N_samples):
        key, subkey = random.split(key)
        if normal:
            coeff = (mu + sigma*random.normal(key, shape=(ord,))) + 1j*(mu + sigma*random.normal(subkey, shape=(ord,)))
        else:
            coeff = (random.uniform(key, shape=(ord,)) - 1/2)*sigma + 1j*(random.uniform(subkey, shape=(ord,)) - 1/2)*sigma
        normalization_factor = jnp.linalg.norm(coeff)
        coeff = coeff / normalization_factor
        f = lambda x: jnp.dot(jnp.array([jnp.exp(1j*jnp.pi*x*k) for k in range(1 + shift, 1 + shift  + ord)]).T, coeff)
        int_f = lambda x: jnp.dot(jnp.array([jnp.exp(1j*jnp.pi*x*k)/(1j*jnp.pi*k) for k in range(1 + shift, 1 + shift  + ord)]).T, coeff)
        features.append(jnp.real(f(x)))
        targets.append(jnp.real(int_f(x)))
    if return_coefficients:
        val_to_coeff = lambda x, periodic=periodic: utils.values_to_coefficients(x.T, periodic=periodic).T
    else:
        val_to_coeff = lambda x: x
    return val_to_coeff(jnp.stack(features)), val_to_coeff(jnp.stack(targets))
