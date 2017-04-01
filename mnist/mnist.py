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
        w = tf.get_variable('w', [shape[-1], output_dim], initializer=tf.random_normal_initializer())
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))

        logits = tf.matmul(input_, w) + bias
        return logits

def softmax_cross_entropy_with_logits(logits, label):
    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label)


X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

logits = linear(X, 10)

# loss function
loss = tf.reduce_mean(softmax_cross_entropy_with_logits(logits, Y))

# optimizer
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# predict & accuracy
prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:

    # Initialize
    sess.run(tf.global_variables_initializer())

    # Training
    for epoch in range(training_epochs):
        avg_loss = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([loss, opt], feed_dict={X: batch_xs, Y: batch_ys})
            avg_loss += c / total_batch

        print('Epoch:', '%04d'%(epoch+1), 'loss =', '{:.9f}'.format(avg_loss))

        # accuracy
        epoch_accuracy = sess.run([accuracy], feed_dict={X: mnist.test.images, Y: mnist.test.labels})
        print('accuracy =', epoch_accuracy)

    total_acc = sess.run([accuracy], feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    print('accuracy =', total_acc)

   
