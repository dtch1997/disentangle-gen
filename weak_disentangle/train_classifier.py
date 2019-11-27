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
from weak_disentangle import metrics

tf.enable_v2_behavior()
tfk = tf.keras


@gin.configurable
def train(dset_name, s_dim, n_dim, factors, s_I_dim,
          batch_size, clas_lr, iterations):
  ut.log("In train classifier")
  # Load data
  dset = datasets.get_dlib_data(dset_name)
  if dset is None:
    x_shape = [64, 64, 1]
  else:
    x_shape = dset.observation_shape
    targets_real = tf.ones((batch_size, 1))
    targets_fake = tf.zeros((batch_size, 1))
    targets = tf.concat((targets_real, targets_fake), axis=0)

  # Networks
  clas = networks.Classifier(x_shape, s_I_dim)
  ut.log(clas.read(clas.WITH_VARS))

  # Create optimizers
  clas_opt = tfk.optimizers.Adam(learning_rate=clas_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-8)

  @tf.function
  def train_clas_step(x_real, y_real):
    clas.train()

    with tf.GradientTape(persistent=True) as tape:
      # Generate
      p_s = clas(x_real)
      clas_loss = -tf.reduce_mean(p_s.log_prob(y_real[:, :s_I_dim]))


    clas_grads = tape.gradient(clas_loss, clas.trainable_variables)
    clas_opt.apply_gradients(zip(clas_grads, clas.trainable_variables))

    return dict(clas_loss=clas_loss)

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
    basedir = os.path.join(FLAGS.basedir, "clas")
    ckptdir = os.path.join(basedir, "ckptdir")
    vizdir = os.path.join(basedir, "vizdir")
    gfile.MakeDirs(basedir)
    gfile.MakeDirs(ckptdir)
    gfile.MakeDirs(vizdir)  # train_range will be specified below

  ckpt_prefix = os.path.join(ckptdir, "model")
  ckpt_root = tf.train.Checkpoint(clas=clas, clas_opt=clas_opt)

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

  # Training
  if dset is None:
    ut.log("Dataset {} is not available".format(dset_name))
    ut.log("Ending program having checked that the networks can be built.")
    return

  batches = datasets.unmasked_label_data_generator(dset, s_dim).repeat().batch(batch_size).prefetch(1000)
  batches = iter(batches)
  start_time = time.time()
  train_time = 0

  if FLAGS.debug:
    train_range = tqdm(train_range)

  for global_step in train_range:
    stopwatch = time.time()
    x, y = next(batches)
    vals = train_clas_step(x, y)
    train_time += time.time() - stopwatch

    # Generic bookkeeping
    if (global_step + 1) % iter_log == 0 or global_step == 0:
      elapsed_time = time.time() - start_time
      string = ", ".join((
          "Iter: {:07d}, Elapsed: {:.3e}, (Elapsed) Iter/s: {:.3e}, (Train Step) Iter/s: {:.3e}".format(
              global_step, elapsed_time, global_step / elapsed_time, global_step / train_time),
          "Clas: {clas_loss:.4f}".format(
          **vals)
      )) + "."
      ut.log(string)

    # Log visualizations and evaluations
    if (global_step + 1) % iter_save == 0 or global_step == 0:
      ut.log("Beginning evaluation.")
      sample_size = 10000
      random_state = np.random.RandomState(1)

      factors = dset.sample_factors(sample_size, random_state)
      obs = dset.sample_observations_from_factors(factors, random_state)

      eval_y = tf.convert_to_tensor(factors, dtype=tf.float32)
      eval_x = tf.convert_to_tensor(obs, dtype=tf.float32)

      p_s_eval = clas(eval_x)
      eval_loss = -tf.reduce_mean(p_s_eval.log_prob(eval_y[:, :s_I_dim]))
      ut.log("Eval loss: {}".format(eval_loss))

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
      "weak_disentangle/configs/clas.gin",
      "Gin bindings to override values in gin config.")
  flags.DEFINE_multi_string(
      "gin_bindings", [],
      "Gin bindings to override values in gin config.")
  app.run(main)
