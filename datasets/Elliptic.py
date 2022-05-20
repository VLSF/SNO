import functions.Fourier as Fourier
import functions.Chebyshev as Chebyshev
import functions.utils as utils
from jax import random
import jax.numpy as jnp

def Fourier_series(ord, n, N_samples, key=random.PRNGKey(1729), generate_rhs=False, shift=0, sigma=1.0, mu=0.0, periodic=False, return_coefficients=True, normal=True):
    if periodic & return_coefficients:
        print("Output does not make any sense for this input")
    features = []
    targets = []
    D = utils.get_differentiation_matrix(n)
    for i in range(N_samples):
        x = utils.grid(n, periodic=False)
        key, subkey = random.split(key)
        if normal:
            coeff = (mu + sigma*random.normal(key, shape=(ord,))) + 1j*(mu + sigma*random.normal(subkey, shape=(ord,)))
        else:
            coeff = (random.uniform(key, shape=(ord,)) - 1/2)*sigma + 1j*(random.uniform(subkey, shape=(ord,)) - 1/2)*sigma
        normalization_factor = jnp.linalg.norm(coeff)
        coeff = coeff / normalization_factor
        f = lambda x: jnp.real(jnp.dot(jnp.array([jnp.exp(1j*jnp.pi*x*k) for k in range(1 + shift, 1 + shift  + ord)]).T, coeff))
        #weight = (1 + f(x)**2)
        weight = 10*(jnp.tanh(f(x)) + 1) + 1
        D2 = -jnp.dot(D, jnp.dot(jnp.diag(weight), D))[1:-1, 1:-1]
        if generate_rhs:
            key, _ = random.split(subkey)
            f = lambda x: jnp.real(jnp.dot(jnp.array([jnp.exp(1j*jnp.pi*x*k) for k in range(1 + shift, 1 + shift  + ord)]).T, coeff))
            rhs = f(x)
        else:
            rhs = jnp.ones(n)
        A = jnp.linalg.inv(D2)
        solution = jnp.dot(A, rhs[1:-1])
        solution = jnp.hstack([0, solution, 0])
        if periodic:
            W = utils.get_interpolation_matrix(jnp.hstack([utils.grid(n-1, periodic=periodic), 1]), n)
            solution = jnp.dot(W, solution) / jnp.dot(W, jnp.ones_like(solution))
            weight = jnp.dot(W, weight.reshape(-1, )) / jnp.dot(W, jnp.ones_like(solution))
            rhs = jnp.dot(W, rhs.reshape(-1, )) / jnp.dot(W, jnp.ones_like(solution))
        if generate_rhs:
            features.append(jnp.vstack([weight.reshape(-1,), rhs.reshape(-1,)]).T)
        else:
            features.append(weight.reshape(-1,))
        targets.append(solution.reshape(-1,))
    if return_coefficients:
        val_to_coeff = lambda x, periodic=periodic: utils.values_to_coefficients(x.T, periodic=periodic).T
    else:
        val_to_coeff = lambda x: x
    if generate_rhs:
        features = jnp.stack(features)
        for i in range(2):
            features = features.at[:, :, i].set(val_to_coeff(features[:, :, i]))
        return features, val_to_coeff(jnp.stack(targets))
    else:
        return val_to_coeff(jnp.stack(features)), val_to_coeff(jnp.stack(targets))


def Fourier_series_2D(ord, n, m, N_samples, key=random.PRNGKey(1729), shift=0, sigma=1.0, mu=0.0, periodic=False, return_coefficients=True):
    if periodic & return_coefficients:
        print("Output does not make any sense for this input")
    features = []
    targets = []
    D1, D2 = utils.get_differentiation_matrix(n), utils.get_differentiation_matrix(m)
    I1, I2 = jnp.eye(n), jnp.eye(m)
    D1, D2 = jnp.kron(I2, D1), jnp.kron(D2, I1)
    x = utils.grid(n, periodic=False)
    y = utils.grid(m, periodic=False)
    x, y = jnp.meshgrid(x, y, indexing="ij")
    zeros_mask = jnp.logical_not(jnp.logical_or(jnp.logical_or((x == 1), (x == -1)), jnp.logical_or((y == 1), (y == -1)))).reshape(-1, )

    for i in range(N_samples):
        keys = random.split(key, 5)
        key = keys[-1]

        coeff1 = (mu + sigma*random.normal(keys[0], shape=(ord,))) + 1j*(mu + sigma*random.normal(keys[1], shape=(ord,)))
        normalization_factor = jnp.linalg.norm(coeff1)
        coeff1 = coeff1 / normalization_factor

        coeff2 = (mu + sigma*random.normal(keys[2], shape=(ord,))) + 1j*(mu + sigma*random.normal(keys[3], shape=(ord,)))
        normalization_factor = jnp.linalg.norm(coeff2)
        coeff2 = coeff2 / normalization_factor

        f = lambda x: jnp.real(jnp.dot(jnp.array([jnp.exp(1j*jnp.pi*x*k) for k in range(1 + shift, 1 + shift  + ord)]).T, coeff1))
        g = lambda x: jnp.real(jnp.dot(jnp.array([jnp.exp(1j*jnp.pi*x*k) for k in range(1 + shift, 1 + shift  + ord)]).T, coeff2))
        weight = ((3*(jnp.tanh(f(x)) + 1) + 1)*(3*(jnp.tanh(g(y)) + 1) + 1)).reshape(-1, )

        A = - jnp.dot(D1, D1 * weight.reshape(-1, 1)) - jnp.dot(D2, D2 * weight.reshape(-1, 1))
        A = A[:, zeros_mask]
        A = A[zeros_mask, :]
        rhs = jnp.ones(A.shape[0])
        res = jnp.dot(jnp.linalg.inv(A), rhs)
        solution = jnp.zeros((n*m))
        solution = solution.at[zeros_mask].set(res)
        solution = solution.reshape(n, m)
        weight = weight.reshape(n, m)
        if periodic:
            W_x = utils.get_interpolation_matrix(jnp.hstack([utils.grid(n-1, periodic=periodic), 1]), n)
            W_y = utils.get_interpolation_matrix(jnp.hstack([utils.grid(m-1, periodic=periodic), 1]), m)
            solution = jnp.dot(W_x, solution) / jnp.dot(W_x, jnp.ones_like(solution))
            solution = jnp.dot(solution, W_y.T) / jnp.dot(jnp.ones_like(solution), W_y.T)
            weight = jnp.dot(W_x, weight) / jnp.dot(W_x, jnp.ones_like(weight))
            weight = jnp.dot(weight, W_y.T) / jnp.dot(jnp.ones_like(weight), W_y.T)
        features.append(weight)
        targets.append(solution)
    if return_coefficients:
        val_to_coeff = lambda x, periodic=periodic: jnp.transpose(utils.values_to_coefficients(jnp.transpose(x, axes=[1, 2, 0]), periodic=periodic), axes=[2, 0, 1])
    else:
        val_to_coeff = lambda x: x
    return val_to_coeff(jnp.stack(features)), val_to_coeff(jnp.stack(targets))
