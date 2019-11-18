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
    boolean_mask = tf.greater(masks[0, :], 0)
    return tf.boolean_mask(x, boolean_mask, axis=1)

def marginalize(p, masks):
    return tfd.MultivariateNormalDiag(loc = mask_reduction(p.mean(), masks),
                                      scale_diag=mask_reduction(p.stddev(), masks)
    )

@gin.configurable
def s_decoder(y_real_pad, gen, enc, masks, k, batch_size, z_dim):
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

    # y_real_pad = y_real # do we need to do padding
    y_real_pad = tf.tile(y_real_pad, [k, 1])
    masks_extended = tf.tile(masks, [k, 1])

    # calculate
    z_fake = datasets.label_randn(batch_size * k, z_dim, masks_extended)
    z_fake = z_fake + y_real_pad
    x_fake = tf.stop_gradient(gen(z_fake))

    p_z = enc(x_fake) # distribs: (batch * k, z_dim)
    means = p_z.mean()
    stds = p_z.stddev()

    p_z_split = []
    p_z_split_I = []
    p_z_split_not_I = []
    for m, s in zip(tf.split(means, k), tf.split(stds, k)):
      p_z_split.append(tfd.MultivariateNormalDiag(loc = m, scale_diag = s))
      p_z_split_I.append(tfd.MultivariateNormalDiag(loc = mask_reduction(m, masks)
        , scale_diag = mask_reduction(s, masks)))
      p_z_split_not_I.append(tfd.MultivariateNormalDiag(loc = mask_reduction(m, 1 - masks)
        , scale_diag = mask_reduction(s, 1 - masks)))

    # p_z_I = marginalize(p_z, masks_extended)
    # p_z_not_I = marginalize(p_z, 1-masks_extended)
    #
    # p_z_split = tf.split(p_z, num_or_size_splits=k) # [distr: (z_dim, batch), k of them]
    # p_z_split_I = tf.split(p_z_I, num_or_size_splits=k)
    # p_z_split_not_I = tf.split(p_z_not_I, num_or_size_splits=k)

    cat = tfd.Categorical(probs=tf.ones((batch_size, k)) / k)
    joint = tfd.Mixture(
        cat=cat,
        components=p_z_split
    ) # distr.shape = (z_dim, batch_size)
    marg_I = tfd.Mixture(
        cat=cat,
        components=p_z_split_I
    ) # distr.shape = (z_dim, batch_size)
    marg_not_I = tfd.Mixture(
        cat=cat,
        components=p_z_split_not_I
    ) # distr.shape = (z_dim, batch_size)

    return joint, marg_I, marg_not_I # each are distribs of shape (z_dim, batch)

@gin.configurable
def mi_estimate(y_real, gen, enc, masks, k, batch_size, z_dim, s_dim,
    z_trans = None):
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

    # taken from config
    y_real_pad = y_real # do we need to do padding

    # decode the s_I's
    joint, marg_I, marg_not_I = s_decoder(y_real_pad, gen, enc, masks, k, batch_size, z_dim)

    # generate z's
    z_fake = datasets.label_randn(batch_size, z_dim, masks)
    z_fake = z_fake + y_real_pad
    # x_fake = tf.stop_gradient(gen(z_fake))

    if z_trans is not None:
      z_fake = z_trans(z_fake)

    z_I_fake = mask_reduction(z_fake, masks)
    z_not_I_fake = mask_reduction(z_fake, 1-masks)

    log_prob_joint = joint.log_prob(z_fake[:, :s_dim])
    log_prob_prod =  marg_I.log_prob(z_I_fake[:, :s_dim]) + marg_not_I.log_prob(z_not_I_fake[:, :s_dim])

    return tf.reduce_mean(log_prob_joint - log_prob_prod)
