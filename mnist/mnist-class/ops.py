import tensorflow as tf

def batch_norm(x, epsilon=1e-5, decay=0.9, train=True):
    return tf.contrib.layers.batch_norm(x,
            decay=decay,
            updates_collections=None,
            epsilon=epsilon,
            scale=True,
            is_training=train)

def linear(input_, output_dim, name='linear'):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('w', [shape[-1], output_dim], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))

        logits = tf.matmul(input_, w) + bias
        return logits

def conv2d(input_, output_dim, filter_w=3, filter_h=3, stride_w=1, stride_h=1, name='conv2d'):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('w', [filter_w, filter_h, shape[-1], output_dim], initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, stride_w, stride_h, 1], padding='SAME')
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))

        return tf.nn.bias_add(conv, bias)

def max_pool(input_, k_w=2, k_h=2):
    return tf.nn.max_pool(input_, ksize=[1, k_w, k_h, 1], strides=[1, k_w, k_h, 1], padding='SAME')

def softmax_cross_entropy_with_logits(logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

def relu(x):
    return tf.nn.relu(x)

