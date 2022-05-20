import jax.numpy as jnp
from jax import random, jit
from functions import utils, Fourier

def F(u, t, k):
    rhs = - 1j*k*Fourier.values_to_coefficients(Fourier.coefficients_to_values(jnp.exp(1j*k**3*t)*u, shape=(2*(u.shape[0]-1), 1))**2)*jnp.exp(-1j*k**3*t)/2
    return rhs

@jit
def RK4(u, k, h, t):
    a = F(u, t, k)
    b = F(u + a*h/2, t + h/2, k)
    c = F(u + b*h/2, t + h/2, k)
    d = F(u + c*h, t + h, k)
    u = u + h*(a + 2*b + 2*c + d)/6
    return u

def generate_trajectory(key, N_solitons=5):
    n = 300
    y = utils.grid(n, periodic=True).reshape(-1, 1)
    keys = random.split(key, 2)
    amplitudes = random.uniform(keys[0], (N_solitons,))*10 + 15
    locations = random.uniform(keys[1], (N_solitons,))*0.6 - 0.3
    u = jnp.zeros(n).reshape(-1, 1)
    for a, loc in zip(amplitudes, locations):
        u += 3*(jnp.cosh(a*(y+loc)/2)/a)**(-2)

    w = Fourier.values_to_coefficients(u)
    k = Fourier.get_frequencies(w.shape[0], is_real=True).reshape(-1, 1) / 1j

    t = 0
    sol = [w, ]
    T = [t, ]

    N_steps = 10000
    h = 0.01/n**2

    for j in range(N_steps):
        w = RK4(w, k, h, t)
        if (j+1)%10 == 0:
            sol.append(w)
            T.append(t)
        t += h

    sol = jnp.hstack(sol)
    sol = jnp.exp(1j*jnp.outer(k**3, jnp.array(T)))*sol
    return sol

def generate_KdV_dataset(N_samples, key, N_solitons=5, prefix=""):
    sol = []
    keys = random.split(key, N_samples)
    for key in keys:
        sol.append(generate_trajectory(key, N_solitons=N_solitons))
    sol = jnp.stack(sol, -1)
    jnp.save(prefix + "Fourier_coeff_KdV_" + str(N_solitons) + "_solitons", sol)

    sol_shape = sol.shape
    sol = sol.reshape((sol_shape[0], sol_shape[1]*sol_shape[2]))
    sol_f = Fourier.coefficients_to_values(sol, shape=(2*(sol.shape[0]-1), sol.shape[1]))
    sol_f = sol_f.reshape((-1, sol_shape[1], sol_shape[2]))
    jnp.save(prefix + "Uniform_grid_KdV_" + str(N_solitons) + "_solitons", sol_f)

    y = utils.grid(sol_f.shape[0])
    W = utils.get_interpolation_matrix_F(y, sol_shape[0], is_real=True)
    sol = jnp.vstack([sol, jnp.conj(sol[1:, :])[::-1, :]])
    sol = jnp.real(jnp.dot(W, sol))
    sol = utils.values_to_coefficients(sol, periodic=False)
    sol = sol.reshape((-1, sol_shape[1], sol_shape[2]))
    jnp.save(prefix + "Chebyshev_coeff_KdV_" + str(N_solitons) + "_solitons", sol)
