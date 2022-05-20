import functions.Fourier as Fourier
import functions.Chebyshev as Chebyshev
import functions.utils as utils
from jax import random
import jax.numpy as jnp

def Kuznetsov_Ma(x, t, nu, x0):
  p = 2*jnp.sqrt(nu**2 - 1)
  omega = p*nu
  psi = (- p**2 * jnp.cos(omega*t) - 2 * 1j * p * nu * jnp.sin(omega * t)) / (2*jnp.cos(omega*t) - 2*nu*jnp.cosh(p*(x - x0))) - 1
  return abs(psi)

def breather(n, m, N_samples, key=random.PRNGKey(1729), periodic=False, return_coefficients=True):
    features = []
    targets = []
    v = 0.5
    for i in range(N_samples):
        x = utils.grid(n, periodic=periodic)
        periodicity_t = False if return_coefficients else periodic
        t = 5 * (utils.grid(m, periodic=periodicity_t) + 1) / 2
        x, t = jnp.meshgrid(x, t, indexing="ij")
        keys = random.split(key, 3)
        key = keys[2]
        nu = random.uniform(keys[0], ) * 2 + 1.5
        features.append(Kuznetsov_Ma(x, t*0, nu, 0))
        targets.append(Kuznetsov_Ma(x, t, nu, 0))
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

def sg_breather(x, t, m, a):
  return jnp.arctan(m/jnp.sqrt(1-m**2)*jnp.sin(jnp.sqrt(1-m**2)*(a*x-t/a))/jnp.cosh(m*(a*x+t/a)))

def sine_gordon_breather(n, m, N_samples, key=random.PRNGKey(1729), periodic=False, return_coefficients=True):
    features = []
    targets = []
    a = 20
    for i in range(N_samples):
        x = utils.grid(n, periodic=periodic)
        periodicity_t = False if return_coefficients else periodic
        t = 50 * (utils.grid(m, periodic=periodicity_t) + 1) / 2
        x, t = jnp.meshgrid(x, t, indexing="ij")
        keys = random.split(key, 3)
        key = keys[2]
        l = random.uniform(keys[0], ) * 0.6 + 0.35
        features.append(sg_breather(x, 0*t, l, a))
        targets.append(sg_breather(x, t, l, a))
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
