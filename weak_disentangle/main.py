# coding=utf-8
# Copyright 2019 The Weak Disentangle Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# python3
"""Main script for all experiments.
"""



import gin
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time
import numpy as np
from absl import app
from absl import flags
from tensorflow import gfile
from tqdm import tqdm

from weak_disentangle import datasets, viz, networks, evaluate
from weak_disentangle import utils as ut
from weak_disentangle import metrics, new_metrics

tf.enable_v2_behavior()
tfk = tf.keras


@gin.configurable
def train(dset_name, s_dim, n_dim, factors, z_transform,
          batch_size, dec_lr, enc_lr_mul, iterations,
          model_type="gen"):

  ut.log("In train")
  masks = datasets.make_masks(factors, s_dim)
  z_dim = s_dim + n_dim
  enc_lr = enc_lr_mul * dec_lr
  z_trans = datasets.get_z_transform(z_transform)
  # Load data
  dset = datasets.get_dlib_data(dset_name)
  # if FLAGS.evaluate:
  #   dset = None
  # else:
  #   dset = datasets.get_dlib_data(dset_name)
  if dset is None:
    x_shape = [64, 64, 1]
  else:
    x_shape = dset.observation_shape
    targets_real = tf.ones((batch_size, 1))
    targets_fake = tf.zeros((batch_size, 1))
    targets = tf.concat((targets_real, targets_fake), axis=0)

  # Networks
  if model_type == "gen":
    assert factors.split("=")[0] in {"c", "s", "cs", "r"}
    y_dim = len(masks)
    dis = networks.Discriminator(x_shape, y_dim)
    gen = networks.Generator(x_shape, z_dim)
    enc = networks.Encoder(x_shape, s_dim)  # Encoder ignores nuisance param
    ut.log(dis.read(dis.WITH_VARS))
    ut.log(gen.read(gen.WITH_VARS))
    ut.log(enc.read(enc.WITH_VARS))
  elif model_type == "van":
    assert factors.split("=")[0] in {"l"}
    dis = networks.LabelDiscriminator(x_shape, s_dim)  # Uses s_dim
    gen = networks.Generator(x_shape, z_dim)
    enc = networks.CovEncoder(x_shape, s_dim)  # Encoder ignores nuisance param
    trans_enc = networks.CovEncoder(x_shape, s_dim)

    clas_path = os.path.join(FLAGS.basedir, "clas")
    clas = networks.Classifier(x_shape, s_dim)
    ckpt_root = tf.train.Checkpoint(clas=clas)
    latest_ckpt = tf.train.latest_checkpoint(clas_path)
    ckpt_root.restore(latest_ckpt)

    ut.log(dis.read(dis.WITH_VARS))
    ut.log(gen.read(gen.WITH_VARS))
    ut.log(enc.read(enc.WITH_VARS))
    ut.log(clas.read(clas.WITH_VARS))

  # Create optimizers
  if model_type in {"gen", "van"}:
    gen_opt = tfk.optimizers.Adam(learning_rate=dec_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-8)
    dis_opt = tfk.optimizers.Adam(learning_rate=enc_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-8)
    enc_opt = tfk.optimizers.Adam(learning_rate=enc_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-8)
    trans_enc_opt = tfk.optimizers.Adam(learning_rate=enc_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-8)

  @tf.function
  def train_gen_step(x1_real, x2_real, y_real):
    gen.train()
    dis.train()
    enc.train()
    # Alternate discriminator step and generator step
    with tf.GradientTape(persistent=True) as tape:
      # Generate
      z1, z2, y_fake = datasets.paired_randn(batch_size, z_dim, masks)
      x1_fake = tf.stop_gradient(gen(z1))
      x2_fake = tf.stop_gradient(gen(z2))

      # Discriminate
      x1 = tf.concat((x1_real, x1_fake), 0)
      x2 = tf.concat((x2_real, x2_fake), 0)
      y = tf.concat((y_real, y_fake), 0)
      logits = dis(x1, x2, y)

      # Encode
      p_z = enc(x1_fake)

      dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=logits, labels=targets))
      # Encoder ignores nuisance parameters (if they exist)
      enc_loss = -tf.reduce_mean(p_z.log_prob(z1[:, :s_dim]))

    dis_grads = tape.gradient(dis_loss, dis.trainable_variables)
    enc_grads = tape.gradient(enc_loss, enc.trainable_variables)

    dis_opt.apply_gradients(zip(dis_grads, dis.trainable_variables))
    enc_opt.apply_gradients(zip(enc_grads, enc.trainable_variables))

    with tf.GradientTape(persistent=False) as tape:
      # Generate
      z1, z2, y_fake = datasets.paired_randn(batch_size, z_dim, masks)
      x1_fake = gen(z1)
      x2_fake = gen(z2)

      # Discriminate
      logits_fake = dis(x1_fake, x2_fake, y_fake)

      gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=logits_fake, labels=targets_real))

    gen_grads = tape.gradient(gen_loss, gen.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grads, gen.trainable_variables))

    return dict(gen_loss=gen_loss, dis_loss=dis_loss, enc_loss=enc_loss)

  @tf.function
  def train_van_step(x_real, y_real, entangle=False):
    gen.train()
    dis.train()
    enc.train()
    trans_enc.train()

    if n_dim > 0:
      padding = tf.zeros((y_real.shape[0], n_dim))
      y_real_pad = tf.concat((y_real, padding), axis=-1)
    else:
      y_real_pad = y_real
    
    if entangle:
        # Alternate discriminator step and generator step
        with tf.GradientTape(persistent=False) as tape:
          # Generate
          dummy_mask = tf.zeros_like(masks)
          z_fake = datasets.paired_randn(batch_size, z_dim, dummy_mask)
          x_fake = gen(z_fake)

          # Discriminate
          logits_fake = dis(x_fake, y_real)

          gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
              logits=logits_fake, labels=targets_real))

        gen_grads = tape.gradient(gen_loss, gen.trainable_variables)
        gen_opt.apply_gradients(zip(gen_grads, gen.trainable_variables))
     
        with tf.GradientTape(persistent=True) as tape:
          # Generate
          dummy_mask = tf.zeros_like(masks)
          z_fake = datasets.paired_randn(batch_size, z_dim, dummy_mask)
          x_fake = tf.stop_gradient(gen(z_fake))
          trans_z_fake = z_trans(z_fake)

          # Discriminate
          x = tf.concat((x_real, x_fake), 0)
          y = tf.concat((y_real, y_real), 0)
          logits = dis(x, y)

          # Encode
          p_z = enc(x_fake)
          p_z_trans = trans_enc(x_fake)

          dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
              logits=logits, labels=targets))
          # Encoder ignores nuisance parameters (if they exist)
          enc_loss = -tf.reduce_mean(p_z.log_prob(z_fake[:, :s_dim]))
          trans_enc_loss = -tf.reduce_mean(p_z_trans.log_prob(
            trans_z_fake[:, :s_dim]))

        dis_grads = tape.gradient(dis_loss, dis.trainable_variables)
        enc_grads = tape.gradient(enc_loss, enc.trainable_variables)
        trans_enc_grads = tape.gradient(trans_enc_loss,
          trans_enc.trainable_variables)

        dis_opt.apply_gradients(zip(dis_grads, dis.trainable_variables))
        enc_opt.apply_gradients(zip(enc_grads, enc.trainable_variables))
        trans_enc_opt.apply_gradients(zip(trans_enc_grads, trans_enc.trainable_variables))

    else:
        if n_dim > 0:
          padding = tf.zeros((y_real.shape[0], n_dim))
          y_real_pad = tf.concat((y_real, padding), axis=-1)
        else:
          y_real_pad = y_real

        # Alternate discriminator step and generator step
        with tf.GradientTape(persistent=False) as tape:
          # Generate
          z_fake = datasets.paired_randn(batch_size, z_dim, masks)
          z_fake = z_fake + y_real_pad
          x_fake = gen(z_fake)

          # Discriminate
          logits_fake = dis(x_fake, y_real)

          gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
              logits=logits_fake, labels=targets_real))

        gen_grads = tape.gradient(gen_loss, gen.trainable_variables)
        gen_opt.apply_gradients(zip(gen_grads, gen.trainable_variables))

        with tf.GradientTape(persistent=True) as tape:
          # Generate
          z_fake = datasets.paired_randn(batch_size, z_dim, masks)
          z_fake = z_fake + y_real_pad
          x_fake = tf.stop_gradient(gen(z_fake))
          trans_z_fake = z_trans(z_fake)

          # Discriminate
          x = tf.concat((x_real, x_fake), 0)
          y = tf.concat((y_real, y_real), 0)
          logits = dis(x, y)

          # Encode
          p_z = enc(x_fake)
          p_z_trans = trans_enc(x_fake)

          dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
              logits=logits, labels=targets))
          # Encoder ignores nuisance parameters (if they exist)
          enc_loss = -tf.reduce_mean(p_z.log_prob(z_fake[:, :s_dim]))
          trans_enc_loss = -tf.reduce_mean(p_z_trans.log_prob(
            trans_z_fake[:, :s_dim]))

        dis_grads = tape.gradient(dis_loss, dis.trainable_variables)
        enc_grads = tape.gradient(enc_loss, enc.trainable_variables)
        trans_enc_grads = tape.gradient(trans_enc_loss,
          trans_enc.trainable_variables)

        dis_opt.apply_gradients(zip(dis_grads, dis.trainable_variables))
        enc_opt.apply_gradients(zip(enc_grads, enc.trainable_variables))
        trans_enc_opt.apply_gradients(zip(trans_enc_grads, trans_enc.trainable_variables))

    return dict(gen_loss=gen_loss, dis_loss=dis_loss, enc_loss=enc_loss, trans_enc_loss=trans_enc_loss)

  @tf.function
  def gen_eval(z):
    gen.eval()
    return gen(z)

  @tf.function
  def enc_eval(x):
    enc.eval()
    return enc(x).mean()
  enc_np = lambda x: enc_eval(x).numpy()

  @tf.function
  def trans_enc_eval(x):
    trans_enc.eval()
    return trans_enc(x).mean()
  trans_enc_np = lambda x: trans_enc_eval(x).numpy()

  # Initial preparation
  if FLAGS.debug:
    iter_log = 100
    iter_save = 2000
    train_range = range(iterations)
    basedir = FLAGS.basedir
    vizdir = FLAGS.basedir
    ckptdir = FLAGS.basedir
    new_run = True
  else:
    iter_log = 5000
    iter_save = 5000
    iter_metric = iter_save * 5  # Make sure this is a factor of 500k
    basedir = os.path.join(FLAGS.basedir, "exp")
    ckptdir = os.path.join(basedir, "ckptdir")
    vizdir = os.path.join(basedir, "vizdir")
    gfile.MakeDirs(basedir)
    gfile.MakeDirs(ckptdir)
    gfile.MakeDirs(vizdir)  # train_range will be specified below

  ckpt_prefix = os.path.join(ckptdir, "model")
  if model_type in {"gen", "van"}:
    ckpt_root = tf.train.Checkpoint(dis=dis, dis_opt=dis_opt,
                                    gen=gen, gen_opt=gen_opt,
                                    enc=enc, enc_opt=enc_opt,
                                    trans_enc=trans_enc, trans_enc_opt=trans_enc_opt)

  # Check if we're resuming training if not in debugging mode
  if not FLAGS.debug:
    latest_ckpt = tf.train.latest_checkpoint(ckptdir)
    if latest_ckpt is None:
      new_run = True
      ut.log("Starting a completely new model")
      train_range = range(iterations)

    else:
      new_run = False
      ut.log("Restarting from {}".format(latest_ckpt))
      ckpt_root.restore(latest_ckpt)
      resuming_iteration = iter_save * (int(ckpt_root.save_counter) - 1)
      train_range = range(resuming_iteration, iterations)

  samples = FLAGS.val_samples
  if FLAGS.evaluate:
    masks = np.zeros([samples, z_dim])
    masks[:, 0] = 1
    masks = tf.convert_to_tensor(masks, dtype=tf.float32)

    transformed_prior = datasets.transformed_prior(z_trans)
    mi, mi_trans, mi_joint, mi_joint_trans = [], [], [], []

    for i in range(FLAGS.mi_averages):
      mi.append(new_metrics.mi_difference(z_dim, gen, clas, masks, samples))
      mi_trans.append(new_metrics.mi_difference(z_dim, gen, clas, masks, samples, z_prior = transformed_prior))

      mi_joint.append(new_metrics.mi_difference(z_dim, gen, clas, masks, samples, draw_from_joint=True))
      mi_joint_trans.append(new_metrics.mi_difference(z_dim, gen, clas, masks, samples, z_prior = transformed_prior, draw_from_joint=True))

    mi = np.mean(np.stack(mi), axis=0)
    mi_trans = np.mean(np.stack(mi_trans), axis=0)

    mi_joint = np.mean(mi_joint)
    mi_joint_trans = np.mean(mi_joint_trans)

    ut.log("MI - Normal: {}, {} Trans: {}, {}".format(mi[0], mi[1], mi_trans[0], mi_trans[1]))
    # mi = new_metrics.mi_difference(z_dim, gen, clas, masks, samples)
    # unmixed_prior = datasets.unmixed_prior(FLAGS.shift, FLAGS.scale)
    # mi_unmixed = new_metrics.mi_difference(z_dim, gen, clas, masks, samples, z_prior = unmixed_prior)
    # mi_mixed = new_metrics.mi_difference(z_dim, gen, clas, masks, samples, z_prior = datasets.mixed_prior)
    # ut.log("MI - Normal: {}, {} Unmixed: {}, {} Mixed: {}, {}".format(mi[0], mi[1], mi_unmixed[0], mi_unmixed[1], mi_mixed[0], mi_mixed[1]))

    # mi_joint = new_metrics.mi_difference(z_dim, gen, clas, masks, samples, draw_from_joint=True)
    # mi_unmixed_joint = new_metrics.mi_difference(z_dim, gen, clas, masks, samples, z_prior = unmixed_prior, draw_from_joint=True)
    # mi_mixed_joint = new_metrics.mi_difference(z_dim, gen, clas, masks, samples, z_prior = datasets.mixed_prior, draw_from_joint=True)
    ut.log("MI Joint - Normal: {} Trans: {}".format(mi_joint, mi_joint_trans))

    ut.log("Encoder Metrics")
    evaluate.evaluate_enc(enc_np, dset, s_dim,
                          FLAGS.gin_file,
                          FLAGS.gin_bindings,
                          pida_sample_size=1000,
                          dlib_metrics=FLAGS.debug_dlib_metrics)
    ut.log("Transformed Encoder Metrics")
    evaluate.evaluate_enc(trans_enc_np, dset, s_dim,
                          FLAGS.gin_file,
                          FLAGS.gin_bindings,
                          pida_sample_size=1000,
                          dlib_metrics=FLAGS.debug_dlib_metrics)

  # Training
  if dset is None:
    ut.log("Dataset {} is not available".format(dset_name))
    ut.log("Ending program having checked that the networks can be built.")
    return

  batches = datasets.paired_data_generator(dset, masks).repeat().batch(batch_size).prefetch(1000)
  batches = iter(batches)
  start_time = time.time()
  train_time = 0

  if FLAGS.debug:
    train_range = tqdm(train_range)
  if FLAGS.visualize:
    train_range = range(iterations+1)

  for global_step in train_range:
    stopwatch = time.time()
    if model_type == "gen":
      x1, x2, y = next(batches)
      vals = train_gen_step(x1, x2, y)
    elif model_type == "van":
      x, y = next(batches)
      vals = train_van_step(x, y, FLAGS.entangle)
    train_time += time.time() - stopwatch

    # Generic bookkeeping
    if (global_step + 1) % iter_log == 0 or global_step == 0:
      elapsed_time = time.time() - start_time
      string = ", ".join((
          "Iter: {:07d}, Elapsed: {:.3e}, (Elapsed) Iter/s: {:.3e}, (Train Step) Iter/s: {:.3e}".format(
              global_step, elapsed_time, global_step / elapsed_time, global_step / train_time),
          "Gen: {gen_loss:.4f}, Dis: {dis_loss:.4f}, Enc: {enc_loss:.4f}, Trans_Enc: {trans_enc_loss:.4f}".format(
          **vals)
      )) + "."
      ut.log(string)

    # Log visualizations and evaluations
    if (global_step + 1) % iter_save == 0 or global_step == 0:
      if model_type == "gen":
        viz.ablation_visualization(x1, x2, gen_eval, z_dim, vizdir, global_step + 1)
      elif model_type == "van":
        viz.ablation_visualization(x, x, gen_eval, z_dim, vizdir, global_step + 1)

      # num_s_I = 100
      # k = 150
      # y_real = tf.convert_to_tensor(dset.sample_factors(num_s_I, np.random.RandomState(1)), dtype=tf.float32)
      masks = np.zeros([samples, z_dim])
      masks[:, 0] = 1
      masks = tf.convert_to_tensor(masks, dtype=tf.float32)
      # y_real = y_real * masks
      # mi = metrics.mi_estimate(y_real, gen, enc, masks, k, num_s_I, z_dim, s_dim)
      # mi_trans = metrics.mi_estimate(y_real, gen, trans_enc, masks, k, num_s_I, z_dim, s_dim, z_trans)
      # ut.log("Encoder MI: {} Transformed Encoder MI: {}".format(mi, mi_trans))
      # mi = new_metrics.mi_difference(z_dim, gen, clas, masks, samples)
      # mi_joint =  new_metrics.mi_difference(z_dim, gen, clas, masks, samples, draw_from_joint=True)
      # ut.log("MI:{} MI_Joint:{}".format(mi, mi_joint))
      mi = new_metrics.mi_difference(z_dim, gen, clas, masks, samples)
      unmixed_prior = datasets.unmixed_prior(FLAGS.shift, FLAGS.scale)
      mi_unmixed = new_metrics.mi_difference(z_dim, gen, clas, masks, samples, z_prior = unmixed_prior)
      mi_mixed = new_metrics.mi_difference(z_dim, gen, clas, masks, samples, z_prior = datasets.mixed_prior)
      ut.log("MI - Normal: {}, {} Unmixed: {}, {} Mixed: {}, {}".format(mi[0], mi[1], mi_unmixed[0], mi_unmixed[1], mi_mixed[0], mi_mixed[1]))

      mi_joint = new_metrics.mi_difference(z_dim, gen, clas, masks, samples, draw_from_joint=True)
      mi_unmixed_joint = new_metrics.mi_difference(z_dim, gen, clas, masks, samples, z_prior = unmixed_prior, draw_from_joint=True)
      mi_mixed_joint = new_metrics.mi_difference(z_dim, gen, clas, masks, samples, z_prior = datasets.mixed_prior, draw_from_joint=True)
      ut.log("MI Joint - Normal: {} Unmixed: {} Mixed: {}".format(mi_joint, mi_unmixed_joint, mi_mixed_joint))


      if FLAGS.debug:
        ut.log("Encoder Metrics")
        evaluate.evaluate_enc(enc_np, dset, s_dim,
                              FLAGS.gin_file,
                              FLAGS.gin_bindings,
                              pida_sample_size=1000,
                              dlib_metrics=FLAGS.debug_dlib_metrics)
        ut.log("Transformed Encoder Metrics")
        evaluate.evaluate_enc(trans_enc_np, dset, s_dim,
                              FLAGS.gin_file,
                              FLAGS.gin_bindings,
                              pida_sample_size=1000,
                              dlib_metrics=FLAGS.debug_dlib_metrics)


      else:
        dlib_metrics = (global_step + 1) % iter_metric == 0
        ut.log("Encoder Metrics")
        evaluate.evaluate_enc(enc_np, dset, s_dim,
                              FLAGS.gin_file,
                              FLAGS.gin_bindings,
                              pida_sample_size=10000,
                              dlib_metrics=dlib_metrics)
        ut.log("Transformed Encoder Metrics")
        evaluate.evaluate_enc(trans_enc_np, dset, s_dim,
                              FLAGS.gin_file,
                              FLAGS.gin_bindings,
                              pida_sample_size=10000,
                              dlib_metrics=dlib_metrics)



    # Save model
    if (global_step + 1) % iter_save == 0 or (global_step == 0 and new_run):
      # Save model only after ensuring all measurements are taken.
      # This ensures that restarts always computes the evals
      ut.log("Saved to", ckpt_root.save(ckpt_prefix))


def main(_):
  if FLAGS.debug:
    FLAGS.gin_bindings += ["log.debug = True"]
  gin.parse_config_files_and_bindings(
      [FLAGS.gin_file],
      FLAGS.gin_bindings,
      finalize_config=False)
  ut.log("\n" + "*" * 80 + "\nBegin program\n" + "*" * 80)
  ut.log("In main")
  train()
  ut.log("\n" + "*" * 80 + "\nEnd program\n" + "*" * 80)


if __name__ == "__main__":
  FLAGS = flags.FLAGS
  flags.DEFINE_string(
      "basedir",
      "savedir/default",
      "Path to directory where to store results.")
  flags.DEFINE_boolean(
      "debug",
      False,
      "Flag debugging mode (shorter run-times, etc)")
  flags.DEFINE_boolean(
      "debug_dlib_metrics",
      False,
      "Flag evaluating dlib metrics when debugging")
  flags.DEFINE_string(
      "gin_file",
      "weak_disentangle/configs/gan.gin",
      "Gin bindings to override values in gin config.")
  flags.DEFINE_multi_string(
      "gin_bindings", [],
      "Gin bindings to override values in gin config.")
  flags.DEFINE_integer(
      "shift",
      1,
      "Translation for unmixed prior")
  flags.DEFINE_integer(
      "scale",
      3,
      "Scaling for unmixed prior")
  flags.DEFINE_boolean(
      "evaluate",
      False,
      "Flag denoting whether to evaluate (for trained models)")
  flags.DEFINE_boolean(
      "visualize",
      False,
      "Flag denoting whether to visualize (for trained models)")
  flags.DEFINE_integer(
      "val_samples",
      100,
      "Number of samples to use in evaluation")
<<<<<<< HEAD
  flags.DEFINE_boolean(
      "entangle",
      False,
      "Set to true to train an entangled model")
=======
  flags.DEFINE_integer(
      "mi_averages",
      10,
      "Number of averages to use in evaluation")
>>>>>>> 3d378910e5013ce2d3b4619d5605ef1fe51563cb
  app.run(main)
