import random
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

def batch_norm(x, epsilon=1e-5, momentum = 0.9, train=True):
    return tf.contrib.layers.batch_norm(x,
            decay=momentum,
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


def softmax_cross_entropy_with_logits(logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

def relu(x):
    return tf.nn.relu(x)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

train = tf.placeholder(tf.bool)

logits1 = relu(batch_norm(linear(X, 256, name='logits1'), train=train))
logits2 = relu(batch_norm(linear(logits1, 256, name='logits2'), train=train))
logits3 = linear(logits2, 10, name='logits3')

loss = tf.reduce_mean(softmax_cross_entropy_with_logits(logits3, Y))

opt = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

prediction = tf.argmax(logits3, 1)
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
            curr_loss, _ = sess.run([loss, opt], feed_dict = {X: batch_xs, Y: batch_ys, train: True})
            avg_loss += curr_loss / total_batch


        print('Epoch:', '%04d'%(epoch+1), 'avg_loss = {:.9f}'.format(avg_loss))

    total_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, train: False})
    print('final accuracy =', '{:.9f}'.format(total_acc))

