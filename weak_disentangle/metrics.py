import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gin
from functools import partial
import functools
from weak_disentangle import datasets, viz, networks, evaluate

tfd = tfp.distributions

def mask_reduction(x, masks):
    # assume all masks are the same.
    boolean_mask = tf.greater(masks[:, 0], 0)
    return tf.boolean_mask(x, boolean_mask)

def marginalize(p, masks):
    return tfd.MultivariateNormalDiag(loc = mask_reduction(p.mean(), masks),
                                      scale_diag=mask_reduction(p.stddev(), masks)
    )

@gin.configurable
def s_decoder(y_real_pad, gen, enc, masks, k):
    """Estimates p(z^_I, z^_\I |s_I), p(z^_I | s_I) and 
    
    For each s_I, we draw k (new) samples of x^.

    Args:
        y_real: s
        gen: generator, (z -> x^)
        enc: encoder, (x^ -> dist(z^))
        masks: for sampling z_\I from the prior
        k: number of samples to estimate with

    Returns:
        joint distr (z^), mariginal distr of z^_I, marginal distr of z^_\I
    """

    # assume we're following label code

    # some variables
    batch_size = 10
    z_dim = 10

    # y_real_pad = y_real # do we need to do padding
    y_real_pad = tf.tile(y_real_pad, k)
    masks_extended = tf.tile(masks, k)

    # calculate
    z_fake = datasets.paired_randn(batch_size * k, z_dim, masks_extended)
    z_fake = z_fake + y_real_pad
    x_fake = tf.stop_gradient(gen(z_fake))

    p_z = enc(x_fake) # distribs: (z_dim, batch * k)
    
    # _, _, masks_idx = z_fake
    # selected_mask = tf.gather(masks, masks_idx) # (z_dim, batch * k)

    p_z_I = marginalize(p_z, masks)
    p_z_not_I = marginalize(p_z, masks)

    p_z_split = tf.split(axis=-1, value=p_z, num_split=k) # [distr: (z_dim, batch), k of them]
    p_z_split_I = tf.split(axis=-1, value=p_z_I, num_split=k)
    p_z_split_not_I = tf.split(axis=-1, value=p_z_not_I, num_split=k)

    joint = tfd.Mixture(
        cat=tfd.Categorical(probs=[1/k for _ in range(k)]),
        components=p_z_split
    ) # distr.shape = (z_dim, batch_size)
    marg_I = tfd.Mixture(
        cat=tfd.Categorical(probs=[1/k for _ in range(k)]),
        components=p_z_split_I
    ) # distr.shape = (z_dim, batch_size)
    marg_not_I = tfd.Mixture(
        cat=tfd.Categorical(probs=[1/k for _ in range(k)]),
        components=p_z_split_not_I
    ) # distr.shape = (z_dim, batch_size)

    return joint, marg_I, marg_not_I # each are distribs of shape (z_dim, batch)

def mi_estimate(y_real, gen, enc, masks, k):
    """Estimates the MI of the encoder.

    Args:
        y_real: s_I
        gen: generator, (z -> x^)
        enc: encoder, (x^ -> dist(z^))
        masks: for sampling z_\I from the prior
        k: number of samples to estimate with

    Returns:
        the averaged MI
    """

    # some variables
    batch_size = 10
    z_dim = 10
    s_dim = None
    
    y_real_pad = y_real # do we need to do padding

    # decode the s_I's
    joint, marg_I, marg_not_I = s_decoder(y_real_pad, gen, enc, masks, k)

    # generate z's # NOTE: I think we have to use the fixed masks here
    z_fake = datasets.paired_randn(batch_size, z_dim, masks)
    z_fake = z_fake + y_real_pad
    # x_fake = tf.stop_gradient(gen(z_fake))
    
    z_I_fake = mask_reduction(z_fake, masks)
    z_not_I_fake = mask_reduction(z_fake, 1-masks)
    
    log_prob_joint = joint.log_prob(z_fake[:, :s_dim])
    log_prob_prod =  marg_I.log_prob(z_I_fake[:, :s_dim]) + marg_not_I.log_prob(z_not_I_fake[:, :s_dim])

    return tf.reduce_mean(log_prob_joint - log_prob_prod)

