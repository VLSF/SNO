import functions.Fourier as Fourier
import functions.Chebyshev as Chebyshev
import functions.utils as utils
from jax import random
import jax.numpy as jnp

def KdV_1D(n, N_samples, key=random.PRNGKey(1729), periodic=False, return_coefficients=True):
    features = []
    targets = []
    f = lambda x, a, t, x0: 3*a**2 / jnp.cosh((a*(((x + x0) + 1 - a**2*t) % 2 - 1))/2)**2
    for i in range(N_samples):
        x = utils.grid(n, periodic=periodic)
        keys = random.split(key, 3)
        key = keys[2]
        a = random.uniform(keys[0], ) * 15 + 10
        x0 = random.uniform(keys[1], ) * 2 - 1
        features.append(f(x, a, 0, x0))
        targets.append(f(x, a, 0.001, x0))
    if return_coefficients:
        val_to_coeff = lambda x, periodic=periodic: utils.values_to_coefficients(x.T, periodic=periodic).T
    else:
        val_to_coeff = lambda x: x
    return val_to_coeff(jnp.stack(features)), val_to_coeff(jnp.stack(targets))

def KdV_1D_solitons(n, N_samples, key=random.PRNGKey(1729), periodic=False, return_coefficients=True):
    features = []
    targets = []
    arg = lambda x, a, t, x0, i: a[i]*(((x + x0[i]) - 4*a[i]**2*t))
    f1 = lambda x, a, t, x0: 2*a[1]**2*(a[0]**2 - a[1]**2)*jnp.sinh(arg(x, a, t, x0, 0))**2
    f2 = lambda x, a, t, x0: 2*a[0]**2*(a[0]**2 - a[1]**2)*jnp.cosh(arg(x, a, t, x0, 1))**2
    d = lambda x, a, t, x0: (a[0]*jnp.cosh(arg(x, a, t, x0, 0))*jnp.cosh(arg(x, a, t, x0, 1)) - a[1]*jnp.sinh(arg(x, a, t, x0, 0))*jnp.sinh(arg(x, a, t, x0, 1)))**2
    for i in range(N_samples):
        x = utils.grid(n, periodic=periodic)
        keys = random.split(key, 3)
        key = keys[2]
        a1 = random.uniform(keys[0], (1,)) * 15 + 10
        chi = random.uniform(keys[1], (1,)) / 2 + 1/2
        a = [a1, a1*chi]
        x0 = [0.6, 0.5]
        t0, t1 = 0, 0.0005
        features.append((f1(x, a, t0, x0) + f2(x, a, t0, x0))/d(x, a, t0, x0))
        targets.append((f1(x, a, t1, x0) + f2(x, a, t1, x0))/d(x, a, t1, x0))
    if return_coefficients:
        val_to_coeff = lambda x, periodic=periodic: utils.values_to_coefficients(x.T, periodic=periodic).T
    else:
        val_to_coeff = lambda x: x
    return val_to_coeff(jnp.stack(features)), val_to_coeff(jnp.stack(targets))

def KdV_2D(n, m, N_samples, key=random.PRNGKey(1729), periodic=False, return_coefficients=True):
    features = []
    targets = []
    f = lambda x, a, t, x0: 3*a**2 / jnp.cosh((a*(((x + x0) + 1 - a**2*t) % 2 - 1))/2)**2
    for i in range(N_samples):
        x = utils.grid(n, periodic=periodic)
        periodicity_t = False if return_coefficients else periodic
        t = 0.001 * (utils.grid(m, periodic=periodicity_t) + 1) / 2
        x, t = jnp.meshgrid(x, t, indexing="ij")
        keys = random.split(key, 3)
        key = keys[2]
        a = random.uniform(keys[0], ) * 15 + 10
        x0 = random.uniform(keys[1], ) * 2 - 1
        features.append(f(x, a, 0*t, x0))
        targets.append(f(x, a, t, x0))
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

def KdV_2D_solitons(n, m, N_samples, key=random.PRNGKey(1729), periodic=False, return_coefficients=True):
    features = []
    targets = []
    arg = lambda x, a, t, x0, i: a[i]*(((x + x0[i]) - 4*a[i]**2*t))
    f1 = lambda x, a, t, x0: 2*a[1]**2*(a[0]**2 - a[1]**2)*jnp.sinh(arg(x, a, t, x0, 0))**2
    f2 = lambda x, a, t, x0: 2*a[0]**2*(a[0]**2 - a[1]**2)*jnp.cosh(arg(x, a, t, x0, 1))**2
    d = lambda x, a, t, x0: (a[0]*jnp.cosh(arg(x, a, t, x0, 0))*jnp.cosh(arg(x, a, t, x0, 1)) - a[1]*jnp.sinh(arg(x, a, t, x0, 0))*jnp.sinh(arg(x, a, t, x0, 1)))**2
    for i in range(N_samples):
        x = utils.grid(n, periodic=periodic)
        periodicity_t = False if return_coefficients else periodic
        t = 0.0005 * (utils.grid(m, periodic=periodicity_t) + 1) / 2
        x, t = jnp.meshgrid(x, t, indexing="ij")
        keys = random.split(key, 3)
        key = keys[2]
        a1 = random.uniform(keys[0], (1,)) * 15 + 10
        chi = random.uniform(keys[1], (1,)) / 2 + 1/2
        a = [a1, a1*chi]
        x0 = [0.6, 0.5]
        features.append((f1(x, a, t*0, x0) + f2(x, a, t*0, x0))/d(x, a, t*0, x0))
        targets.append((f1(x, a, t, x0) + f2(x, a, t, x0))/d(x, a, t, x0))
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
