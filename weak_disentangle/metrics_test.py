import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gin
from functools import partial
import functools
from weak_disentangle import datasets, viz, networks, evaluate, new_metrics_redux

from weak_disentangle import tensorsketch as ts
from weak_disentangle import utils as ut

z_dim = 2
s_I_dim = 1
samples = 1000

masks = np.zeros([samples, z_dim])
masks[:, 0] = 1
masks = tf.convert_to_tensor(masks, dtype=tf.float32)

def test_gen(z):
    # identity function
    return z

def test_clas(x):
    a, b = tf.split(x, [s_I_dim, z_dim-s_I_dim], axis=-1)
    return tfd.MultivariateNormalDiag(
        loc=a,
        scale_diag=1e-8)

z_trans = datasets.get_z_transform(z_transform)
transformed_prior = datasets.transformed_prior(z_trans)

MI = new_metrics_redux.mi_difference(z_dim, test_gen, test_clas, masks, samples)
MI_transformed = new_metrics_redux.mi_difference(z_dim, test_gen, test_clas, masks, samples, z_prior = transformed_prior)
print(MI)


