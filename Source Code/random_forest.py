# random forest in tensorflow
"""A stand-alone example for tf.learn's random forest model on mnist."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import tensorflow as tf

# pylint: disable=g-backslash-continuation
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.tensor_forest.client import eval_metrics
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import app

FLAGS = None
use_training_loss=False

tf.logging.set_verbosity(tf.logging.DEBUG)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1)
config = tf.ConfigProto(allow_soft_placement=True,gpu_options = gpu_options)

sess = tf.InteractiveSession(config=config)


def build_estimator(model_dir):
  """Build an estimator."""
  params = tensor_forest.ForestHParams(
      num_classes=10, num_features=784,
      num_trees=100, max_nodes=1000)
  graph_builder_class = tensor_forest.RandomForestGraphs
  if use_training_loss:
    graph_builder_class = tensor_forest.TrainingLossForest
  # Use the SKCompat wrapper, which gives us a convenient way to split
  # in-memory data like MNIST into batches.
  return estimator.SKCompat(random_forest.TensorForestEstimator(
      params, graph_builder_class=graph_builder_class,
      model_dir=model_dir))


def train_and_eval():
  """Train and evaluate the model."""

  model_dir = 'data/model'
  print('model directory = %s' % model_dir)

  est = build_estimator(model_dir)

  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  with tf.device('/gpu:0'):
      est.fit(x=mnist.train.images, y=mnist.train.labels,
              batch_size=100,steps=10)

      # results2=est.predict(x=mnist.test.images, y=mnist.test.labels, batch_size=100)
      # print(results2)

      metric_name = 'accuracy'
      metric = {metric_name:
                metric_spec.MetricSpec(
                    eval_metrics.get_metric(metric_name),
                    prediction_key=eval_metrics.get_prediction_key(metric_name))}

      results = est.score(x=mnist.test.images, y=mnist.test.labels,
                          batch_size=100)
      for key in sorted(results):
        print('%s: %s' % (key, results[key]))



train_and_eval()
