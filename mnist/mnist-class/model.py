import tensorflow as tf
import os

from ops import *

from tensorflow.examples.tutorials.mnist import input_data

class MNIST(object):
    def __init__(self, sess, input_height=28, input_width=28, checkpoint_dir=None, name='mnist'):

        self.sess = sess
        self.input_height=input_height
        self.input_width=input_width

        self.name=name
        self.checkpoint_dir=checkpoint_dir

        self.build_model()

    def build_model(self):
        with tf.variable_scope(self.name):

            self.X = tf.placeholder(tf.float32, [None, self.input_width * self.input_height]) # 28 * 28
            X_img = tf.reshape(self.X, [-1, self.input_width, self.input_height, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            self.train = tf.placeholder(tf.bool)
            self.p_conv = tf.placeholder(tf.float32)
            self.p_hidden = tf.placeholder(tf.float32)

            conv1 = relu(batch_norm(conv2d(X_img, 32, name='conv1'), train=self.train))
            conv1 = max_pool(conv1)
            conv1 = tf.nn.dropout(conv1, keep_prob=self.p_conv)

            conv2 = relu(batch_norm(conv2d(conv1, 64, name='conv2'), train=self.train))
            conv2 = max_pool(conv2)
            conv2 = tf.nn.dropout(conv2, keep_prob=self.p_conv)

            conv3 = relu(batch_norm(conv2d(conv2, 128, name='conv3'), train=self.train))
            conv3 = max_pool(conv3)
            conv3 = tf.nn.dropout(conv3, keep_prob=self.p_conv)

            conv4 = relu(batch_norm(conv2d(conv3, 256, name='conv4'), train=self.train))
            conv4 = max_pool(conv4)
            conv4 = tf.nn.dropout(conv4, keep_prob=self.p_conv)

            fc1 = tf.reshape(conv4, [-1, 2*2*256])
            fc1 = relu(batch_norm(linear(fc1, 200, name='fc1'), train=self.train))
            fc1 = tf.nn.dropout(fc1, keep_prob=self.p_hidden)

            self.logits = linear(fc1, 10, name='logits')

            self.loss = tf.reduce_mean(softmax_cross_entropy_with_logits(self.logits, self.Y))

            self.saver = tf.train.Saver()


    def trainmodel(self, config, p_conv=0.8, p_hidden=0.6):
        mnist=input_data.read_data_sets("../MNIST_data", one_hot=True)
            
        self.opt = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        
        self.sess.run(tf.global_variables_initializer())

        loaded, global_counter = self.load(self.checkpoint_dir)
        if loaded:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        for epoch in range(config.training_epochs):
            total_batch = int(mnist.train.num_examples / config.batch_size)
            avg_loss = 0

            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(config.batch_size)
                curr_loss, _ = self.sess.run([self.loss, self.opt], feed_dict={self.X: batch_xs, self.Y: batch_ys, self.train: True, self.p_conv: 0.8, self.p_hidden: 0.6})
                avg_loss += curr_loss / total_batch

                global_counter+=1

            self.save(self.checkpoint_dir, global_counter)
            print('Epoch:', '%04d'%(epoch+1), 'avg_loss = {:.9f}'.format(avg_loss))

    def get_accuracy(self):
        mnist=input_data.read_data_sets("../MNIST_data", one_hot=True)
        
        prediction = tf.argmax(self.logits, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        total_acc = self.sess.run(self.accuracy, feed_dict={self.X: mnist.test.images, self.Y: mnist.test.labels, self.train: False, self.p_conv: 1, self.p_hidden: 1})
        print('final accuracy =', '{:.9f}'.format(total_acc))


    def save(self, checkpoint_dir, step):
        model_name = 'MNIST.model'
#        checkpoint_dir = os.path.join(checkpoint_dir, "MNIST")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            try:
                ckpt_name = ckpt.model_checkpoint_path
                self.saver.restore(self.sess, ckpt_name)
                counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
                print(" [*] Success to read {}".format(ckpt_name))
                return True, counter
            except:
                print(" [*] Failed to find a checkpoint")
                return False, 0
