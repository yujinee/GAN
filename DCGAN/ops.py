import math
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops

from utils import *

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def linear(input_, output_dim, name="linear", stddev=0.02, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', [shape[1], output_dim], tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))

        if with_w:
            return tf.matmul(input_, w), w, bias
        return tf.matmul(input_, w) + bias


def relu(x):
    return tf.nn.relu(x)

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

def conv2d(input_, output_dim, f_h=5, f_w=5, s_h=2, s_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        shape = input_.get_shape().as_list()
        w = tf.get_variable('w', [f_h, f_w, shape[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev = stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, s_h, s_w, 1], padding='SAME')

        bias = tf.get_variable('bias', [output_dim],
                               initializer=tf.constant_initializer(0.0))
        return tf.nn.bias_add(conv, bias)

def deconv2d(input_, output_shape, f_h=5, f_w=5, s_h=2, s_w=2, stddev=0.02, name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        shape = input_.get_shape().as_list()
        w = tf.get_variable('w', [f_h, f_w, output_shape[-1], shape[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, s_h, s_w, 1])

        bias = tf.get_variable('bias', [output_shape[-1]],
                               initializer=tf.constant_initializer(0.0))

        if with_w:
            return tf.nn.bias_add(deconv, bias), w, bias

        return tf.nn.bias_add(deconv, bias)
