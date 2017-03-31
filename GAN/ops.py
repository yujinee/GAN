import numpy as np
import tensorflow as tf

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim /2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def matmul(input_, output_dim, name="matmul"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', initializer=xavier_init([input_.get_shape().as_list()[-1], output_dim]))
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, w) + biases

def relu(input_):
    return tf.nn.relu(input_)

def sigmoid(input_):
    return tf.nn.sigmoid(input_)

def sigmoid_cross_entropy_with_logits(input_, label):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=input_, labels=label)

def getloss(input_):
    return tf.reduce_mean(input_)

def adamOptMinimize(input_, var_list):
    return tf.train.AdamOptimizer().minimize(input_, var_list=var_list)
