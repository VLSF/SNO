import jax.numpy as jnp
from jax import random, jit
from functions import utils, Fourier

def F(u, t, nu):
    rhs = -Fourier.transform_coefficients(Fourier.values_to_coefficients(Fourier.coefficients_to_values(u, shape=(2*(u.shape[0]-1), 1))**2), 0, "diff")/2
    rhs = rhs + nu*Fourier.transform_coefficients(Fourier.transform_coefficients(u, 0, "diff"), 0, "diff")
    return rhs

@jit
def RK4(u, nu, h, t):
    a = F(u, t, nu)
    b = F(u + a*h/2, t + h/2, nu)
    c = F(u + b*h/2, t + h/2, nu)
    d = F(u + c*h, t + h, nu)
    u = u + h*(a + 2*b + 2*c + d)/6
    return u

def generate_trajectory(key, smoothness="high", cherry_pick=False, periodic=True, return_coeff=True):
    n, mu, sigma, ord, h, t, N_steps = 300, 0, 1.0, 20, 0.0001, 0, 10000
    nu = 0.1 if smoothness == "high" else 0.01
    key1, key2 = random.split(key)
    coeff = (mu + sigma*random.normal(key1, shape=(ord,))) + 1j*(mu + sigma*random.normal(key2, shape=(ord,)))
    coeff = coeff / jnp.linalg.norm(coeff)
    f = lambda x: jnp.real(jnp.dot(jnp.vstack([jnp.exp(1j*jnp.pi*i*x) for i in range(ord)]).T, coeff))
    x = utils.grid(n, periodic=True)
    u = f(x).reshape(-1, 1)
    w = Fourier.values_to_coefficients(u)

    sol = [w, ]
    for j in range(N_steps):
        w = RK4(w, nu, h, t)
        if (j+1)%10 == 0:
            sol.append(w)
        t += h

    sol = jnp.hstack(sol)
    quality = jnp.max(abs(sol[-10:, :]))
    if cherry_pick:
        n_less_smooth = jnp.argmax(jnp.max(abs(sol[-10:, :]), axis=0))
        l = max([n_less_smooth-100, 0])
        r = min([n_less_smooth+100, N_steps+1])
        sol = sol[:, l:r]

    if not periodic:
        y = utils.grid(n)
        W = utils.get_interpolation_matrix_F(y, sol.shape[0], is_real=True)
        sol = jnp.vstack([sol, jnp.conj(sol[1:, :])[::-1, :]])
        sol = jnp.real(jnp.dot(W, sol))
        if return_coeff:
            sol = utils.values_to_coefficients(sol, periodic=periodic)
        return sol
    else:
        if not return_coeff:
            sol = Fourier.coefficients_to_values(sol, shape=(2*(w.shape[0]-1), N_steps+1))
        return sol

def generate_Burgers_dataset(N_samples, key, smoothness="high", prefix=""):
    sol = []
    keys = random.split(key, N_samples)
    for key in keys:
        sol.append(generate_trajectory(key, smoothness=smoothness))
    sol = jnp.stack(sol, -1)
    jnp.save(prefix + "Fourier_coeff_Burgers_" + smoothness + "_smoothness", sol)
