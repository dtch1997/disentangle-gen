import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gin
from functools import partial
import functools
from weak_disentangle import datasets, viz, networks, evaluate

tfd = tfp.distributions

def parallel_encode_into_s(z_I, gen, clas, masks, k=100, lock_samples=True, z_notI=None):
    """
    Estimates p(s_I | z_I) based on sampling.

    Args:
        z_I: z_I (same dims as z)
        gen: generator
        clas: classifier
        masks: indices
        k: sample size
        lock_samples: whether to redraw samples for each batch item
        z_notI: fix samples. This assumes samples are locked, size: (batch_size, z_dim)

    Returns:
        s_dist, z_notI
    """
    batch_size, z_dim = z_I.shape

    z_I_extended = tf.tile(z_I, [k, 1])
    masks_extended = tf.tile(masks, [k, 1])
    z_notI_extended = None

    if z_notI == None:
        if lock_samples:
            z_notI = datasets.label_randn(batch_size, z_dim, masks)
        else:
            z_notI_extended = datasets.label_randn(batch_size * k, z_dim, masks_extended)
    
    if z_notI_extended == None:
        z_notI_extended = tf.tile(z_notI, [k, 1])

    z = z_I_extended + z_notI_extended

    x_hat = tf.stop_gradient(gen(z))

    p_s = clas(x_hat) #  distribs: (batch * k, s_dim = zI_dim)
    means = p_s.mean()
    stds = p_s.stddev()

    p_s_split = [tfd.MultivariateNormalDiag(loc = m, scale_diag = s) for m, s in zip(tf.split(means, k), tf.split(stds, k))]
    cat = tfd.Categorical(probs=tf.ones((batch_size, k)) / k)

    s_dist = tfd.Mixture(
        cat=cat,
        components=p_s_split
    ) # distr.shape = (batch_size, z_dim)

    return s_dist, z_notI

def p_s(s_I, z_dim, gen, clas, masks, k=100, z_notI = None):
    """
    Estimates p(s_I) based on sampling. Batched

    Args:
        s_I: features
        z_dim: dimension of latent space
        gen: generator
        clas: classifier
        masks: indices
        k: sample_size
        z_notI: (defaults to None) if you want to lock z_notI. Otherwise, will draw from joint.

    Returns:

    """
    batch_size, s_dim = s_I.shape

    if z_notI == None:
        # generate
        blank_mask = tf.zeros([batch_size, z_dim])
        z = datasets.label_randn(batch_size, z_dim, blank_mask)
        x_hat = tf.stop_gradient(gen(z))

        p_s = clas(x_hat) #  distribs: (batch * k, s_dim = z_dim)
        return tf.reduce_mean(p_s.prob(s_I))
    else:
        z_I = datasets.label_randn(batch_size, z_dim, 1-masks)
        p_s_zI = parallel_encode_into_s(z_I, gen, clas, masks, z_notI=z_notI)
        return tf.reduce_mean(p_s_zI.prob(s_I))

def mi_estimate(z_dim, gen, clas, masks, batch_size, k=100):
    z_I = datasets.label_randn(batch_size, z_dim, 1-masks)
    s_distr, z_notI = parallel_encode_into_s(z_I, gen, clas, masks, k=k, lock_samples=True)
    s_I = s_distr.sample()
    logp_siz = s_distr.log_prob(s_I)
    logp_si = tf.log(p_s(s_I, z_dim, gen, clas, masks, z_notI=z_notI))
    return tf.reduce_mean(logp_siz - logp_si)

def mi_difference(z_dim, gen, clas, masks, batch_size, k=100, draw_from_joint=False):
    if draw_from_joint:
        blank_mask = tf.zeros([batch_size, z_dim])
        z = datasets.label_randn(batch_size, z_dim, blank_mask)
        s_I = clas(tf.stop_gradient(gen(z))).sample()
        z_I = z * masks
        z_notI = z * (1-masks)

        p_s_zI = parallel_encode_into_s(z_I, gen, clas, masks, k=k, lock_samples=False)
        p_s_znotI = parallel_encode_into_s(z_notI, gen, clas, masks, k=k, lock_samples=False)

        return tf.reduce_mean(p_s_zI.log_prob(s_I) - p_s_znotI.log_prob(s_I))
    else:
        I_zI = mi_estimate(z_dim, gen,clas, masks, batch_size, k=k)
        I_znotI = mi_estimate(z_dim, gen, clas, 1-masks, batch_size, k=k)
        return I_zI - I_znotI
