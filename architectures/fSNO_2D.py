import functions.utils as utils
import functions.Chebyshev as Chebyshev
import functions.Fourier as Fourier
import jax.numpy as jnp
import os

from jax import random, jit, grad, vmap
from functools import partial

activation_function = utils.complex_split_softplus

def random_c_layer_params(m, n, key):
    keys = random.split(key, 4)
    layer_parameters = (random.normal(keys[0], (m, n)) / m, random.normal(keys[1], (m, n)) / m, random.normal(keys[1], (1, n)), random.normal(keys[3], (1, 1, n)))
    return layer_parameters

def init_c_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_c_layer_params(m, n, r) for m, n, r in zip(sizes[:-1], sizes[1:], keys)]

@partial(jit, static_argnums=2)
def NN_c(params, input, activation):
    n = len(params)
    for i, p in enumerate(params):
        input = jnp.dot(input, p[0] + 1j * p[1]) + p[2] + 1j * p[3]
        if i < n-1:
            input = activation(input)
    return input

def random_i_layer_params(m_x, n_x, m_y, n_y, c_m, c_n, key):
    keys = random.split(key, 7)
    layer_parameters = (random.normal(keys[0], (n_x, m_x)) / m_x, random.normal(keys[1], (n_x, m_x)) / m_x, random.normal(keys[2], (c_m, c_n)) / c_m, random.normal(keys[3], (c_m, c_n)) / c_m, random.normal(keys[4], (n_x, n_y, c_n)), random.normal(keys[5], (n_x, n_y, c_n)), random.normal(keys[6], (n_y, m_y)) / m_y)
    return layer_parameters

def init_i_network_params(sizes_x, sizes_y, c_sizes, key):
    keys = random.split(key, len(sizes_x))
    return [random_i_layer_params(m_x, n_x, m_y, n_y, c_m, c_n, k) for m_x, n_x, m_y, n_y, c_m, c_n, k in zip(sizes_x[:-1], sizes_x[1:], sizes_y[:-1], sizes_y[1:], c_sizes[:-1], c_sizes[1:], keys)]

def NN_i(params, input, activation):
    n = len(params)
    for i, p in enumerate(params):
        input = jnp.dot(p[0] + 1j * p[1], jnp.dot(p[6], jnp.dot(input, p[2] + 1j * p[3]))) + p[4] + 1j * p[5]
        if i < n-1:
            input = activation(input)
    return input

@partial(jit, static_argnums=2)
def NN(params, input):
    input = NN_c(params[0], input, activation_function)
    input = NN_i(params[1], input, activation_function)
    input = NN_c(params[2], input, activation_function)
    return input

batched_NN = vmap(NN, in_axes=(None, 0))
batched_norm = vmap(lambda x: jnp.linalg.norm(x.reshape(-1, )), in_axes=(0,))

@jit
def loss(params, x, y):
    return jnp.mean(batched_norm(batched_NN(params, x) - y))

def save_params(params, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = filename if filename[-4:] == ".npz" else filename + ".npz"

    params_encoder_dict = {}
    for i, p in enumerate(params[0]):
        params_encoder_dict["w_r_" + str(i)] = p[0]
        params_encoder_dict["w_c_" + str(i)] = p[1]
        params_encoder_dict["b_r_" + str(i)] = p[2]
        params_encoder_dict["b_c_" + str(i)] = p[3]
    with open(directory + "/" + "encoder_" + filename, "wb") as f:
        jnp.savez(f, **params_encoder_dict)

    params_i_dict = {}
    for i, p in enumerate(params[1]):
        params_i_dict["w_r_" + str(i)] = p[0]
        params_i_dict["w_c_" + str(i)] = p[1]
        params_i_dict["s_r_" + str(i)] = p[2]
        params_i_dict["s_c_" + str(i)] = p[3]
        params_i_dict["b_r_" + str(i)] = p[4]
        params_i_dict["b_c_" + str(i)] = p[5]
        params_i_dict["v_" + str(i)] = p[6]
    with open(directory + "/" + "i_" + filename, "wb") as f:
        jnp.savez(f, **params_i_dict)

    params_decoder_dict = {}
    for i, p in enumerate(params[2]):
        params_decoder_dict["w_r_" + str(i)] = p[0]
        params_decoder_dict["w_c_" + str(i)] = p[1]
        params_decoder_dict["b_r_" + str(i)] = p[2]
        params_decoder_dict["b_c_" + str(i)] = p[3]
    with open(directory + "/" + "decoder_" + filename, "wb") as f:
        jnp.savez(f, **params_decoder_dict)

def load_params(directory, filename):
    filename = filename if filename[-4:] == ".npz" else filename + ".npz"

    params_encoder_dict = jnp.load(directory + "/" + "encoder_" + filename)
    n = len(params_encoder_dict.files) // 4
    params_encoder = []
    for i in range(n):
        params_encoder.append((params_encoder_dict["w_r_" + str(i)], params_encoder_dict["w_c_" + str(i)], params_encoder_dict["b_r_" + str(i)], params_encoder_dict["b_c_" + str(i)]))

    params_i_dict = jnp.load(directory + "/" + "i_" + filename)
    n = len(params_i_dict.files) // 7
    params_i = []
    for i in range(n):
        params_i.append((params_i_dict["w_r_" + str(i)], params_i_dict["w_c_" + str(i)], params_i_dict["s_r_" + str(i)], params_i_dict["s_c_" + str(i)], params_i_dict["b_r_" + str(i)], params_i_dict["b_c_" + str(i)], params_i_dict["v_" + str(i)]))

    params_decoder_dict = jnp.load(directory + "/" + "decoder_" + filename)
    n = len(params_decoder_dict.files) // 4
    params_decoder = []
    for i in range(n):
        params_decoder.append((params_decoder_dict["w_r_" + str(i)], params_decoder_dict["w_c_" + str(i)], params_decoder_dict["b_r_" + str(i)], params_decoder_dict["b_c_" + str(i)]))

    return [params_encoder, params_i, params_decoder]

def count_params(params):
    count = lambda x: sum([len(q.reshape(-1, )) for q in x])
    return sum([sum([count(q) for q in p]) for p in params])
