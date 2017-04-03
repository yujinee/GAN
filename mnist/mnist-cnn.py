import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

def linear(input_, output_dim, name='linear'):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('w', [shape[-1], output_dim], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, w) + bias

def conv2d(input_, output_dim, filter_w=3, filter_h=3, stride_w=1, stride_h=1, stddev=0.02, name='conv2d'):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('w', [filter_w, filter_h, shape[-1], output_dim], initializer=tf.random_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, stride_w, stride_h, 1], padding='SAME')
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, bias)
        return conv

def max_pool(input_, k_w=2, k_h=2):
    return tf.nn.max_pool(input_, ksize=[1, k_w, k_h, 1], strides=[1, k_w, k_h, 1], padding='SAME')

def softmax_cross_entropy_with_logits(logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

def relu(x):
    return tf.nn.relu(x)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
X_img = tf.reshape(X, [-1, 28, 28, 1])

p_conv = tf.placeholder(tf.float32)
p_hidden = tf.placeholder(tf.float32)

# (?, 28, 28, 1)
conv1 = relu(conv2d(X_img, 32, name='conv1'))
conv1 = max_pool(conv1)
conv1 = tf.nn.dropout(conv1, p_conv)

# (?, 14, 14, 32)
conv2 = relu(conv2d(conv1, 64, name='conv2'))
conv2 = max_pool(conv2)
conv2 = tf.nn.dropout(conv2, p_conv)

# (?, 7, 7, 64)
conv3 = relu(conv2d(conv2, 128, name='conv3'))
conv3 = max_pool(conv3)
conv3 = tf.nn.dropout(conv3, p_conv)

# (?, 4, 4, 128)
fc1 = tf.reshape(conv3, [-1, 4*4*128])
fc1 = relu(linear(fc1, 625, name='fc1'))
fc1 = tf.nn.dropout(fc1, p_hidden)

# 625 -> 10
logits = linear(fc1, 10, name='logits')

loss = tf.reduce_mean(softmax_cross_entropy_with_logits(logits, Y))

opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples / batch_size)
        avg_loss = 0

        print('total_batch : ', total_batch)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            curr_loss, _ = sess.run([loss, opt], feed_dict = {X: batch_xs, Y: batch_ys, p_conv:0.8, p_hidden:0.5})
            avg_loss += curr_loss / total_batch


        print('Epoch:', '%04d'%(epoch+1), 'avg_loss = {:.9f}'.format(avg_loss))

    total_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, p_conv:1, p_hidden: 1})
    print('final accuracy =', '{:.9f}'.format(total_acc))

