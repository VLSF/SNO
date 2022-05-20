import functions.utils as utils
import jax.numpy as jnp
import os

from jax import random, jit, grad, vmap
from functools import partial

def random_layer_params(m_x, n_x, m_y, n_y, k, l, key):
    keys = random.split(key, 5)
    layer_parameters = (random.normal(keys[0], (n_x, m_x)) / m_x, random.normal(keys[1], (m_y, n_y)) / m_y, random.normal(keys[2], (n_x, n_y)), random.normal(keys[3], (k, l)) / k, random.normal(keys[4], (l, )))
    return layer_parameters

def init_network_params(sizes_x, sizes_y, c_sizes, key):
    keys = random.split(key, len(sizes_x)+1)
    return [random_layer_params(m_x, n_x, m_y, n_y, k, l, r) for m_x, n_x, m_y, n_y, k, l, r in zip(sizes_x[:-1], sizes_x[1:], sizes_y[:-1], sizes_y[1:], c_sizes[:-1], c_sizes[1:], keys)] + [(random.normal(keys[-1], (1, 1)))]

def xNN(params, input, activation=jnp.tanh):
    n = len(params) - 1
    for i, p in enumerate(params[:-1]):
        input = jnp.dot(jnp.dot(p[0], input), p[1]) + p[2]
        if i < n-1:
            input = activation(input)
    return input

def cNN(params, input, activation=jnp.tanh):
    n = len(params) - 1
    for i, p in enumerate(params[:-1]):
        input = jnp.dot(input, p[3]) + p[4]
        if i < n-1:
            input = activation(input)
    return input

def NN(params, input_v, input_x, activation=jnp.tanh):
    v = xNN(params, input_v, activation=activation)
    x = cNN(params, input_x, activation=activation)
    return jnp.dot(x, v.reshape(-1,)) + params[-1][0]

batched_NN = vmap(NN, in_axes=(None, 0, None))
batched_norm = vmap(lambda x: jnp.linalg.norm(x.reshape(-1, )), in_axes=(0,))

@jit
def loss(params, v, x, y):
    return jnp.mean(batched_norm(batched_NN(params, v, x) - y))

def save_params(params, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = filename if filename[-4:] == ".npz" else filename + ".npz"

    params_dict = {}
    for i, p in enumerate(params[:-1]):
        params_dict["w_b_x_" + str(i)] = p[0]
        params_dict["w_b_y_" + str(i)] = p[1]
        params_dict["b_b_" + str(i)] = p[2]
        params_dict["w_t_" + str(i)] = p[3]
        params_dict["b_t_" + str(i)] = p[4]
    params_dict["bias"] = params[-1][0]
    with open(directory + "/" + filename, "wb") as f:
        jnp.savez(f, **params_dict)

def load_params(directory, filename):
    filename = filename if filename[-4:] == ".npz" else filename + ".npz"

    params_dict = jnp.load(directory + "/" + filename)
    n = (len(params_dict.files)-1) // 5
    params = []
    for i in range(n):
        params.append((params_dict["w_b_x_" + str(i)], params_dict["w_b_y_" + str(i)], params_dict["b_b_" + str(i)], params_dict["w_t_" + str(i)], params_dict["b_t_" + str(i)]))
    params.append((params_dict["bias"], ))
    return params

def count_params(params):
    count = lambda x: sum([len(q.reshape(-1, )) for q in x])
    return sum([count(q) for q in params])
