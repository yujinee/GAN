import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from ops import *


X = tf.placeholder(tf.float32, shape=[None, 784], name='x')
Z = tf.placeholder(tf.float32, shape=[None, 100], name='z')

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m,n])

def generator(z, name='generator'):
    with tf.variable_scope(name):
        G_h1 = relu(matmul(z, 128, name="g_h1"))
        G_logit = sigmoid(matmul(G_h1, 784, name="g_logit"))
        return G_logit

def discriminator(x, name='discriminator', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        D_h1 = relu(matmul(x, 128, name='d_h1'))
        D_logit = matmul(D_h1, 1, name='d_logit')
        return D_logit

G_sample = generator(Z)
D_logit_real = discriminator(X)
D_logit_fake = discriminator(G_sample, reuse=True)

t_vars = tf.trainable_variables()

theta_D = [var for var in t_vars if 'd_' in var.name]
theta_G = [var for var in t_vars if 'g_' in var.name]

D_loss_real = getloss(sigmoid_cross_entropy_with_logits(D_logit_real, tf.ones_like(D_logit_real)))
D_loss_fake = getloss(sigmoid_cross_entropy_with_logits(D_logit_fake, tf.zeros_like(D_logit_fake)))

D_loss = D_loss_real + D_loss_fake
G_loss = getloss(sigmoid_cross_entropy_with_logits(D_logit_fake, tf.ones_like(D_logit_fake)))

D_solver = adamOptMinimize(D_loss, theta_D)
G_solver = adamOptMinimize(G_loss, theta_G)

batch_size = 128
Z_dim = 100

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

i = 0

for it in range(100000):
    if it%1000 == 0 :
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
        
        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i+=1
        plt.close(fig)
    

    X_mb, _ = mnist.train.next_batch(batch_size)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, Z_dim)})
        
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})
    
    if it%1000 == 0:
        print('iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G loss: {:.4}'.format(G_loss_curr))
        print()

