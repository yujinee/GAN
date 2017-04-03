import tensorflow as tf

from model import *

flags = tf.app.flags
flags.DEFINE_integer("training_epochs", 15, "Epoches to train [15]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for Adam Optimizer [0.001]")
flags.DEFINE_integer("batch_size", 100, "Batch Size [100]")
flags.DEFINE_integer("input_width", 28, "Input image width [28]")
flags.DEFINE_integer("input_height", 28, "Input image height [28]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
FLAGS = flags.FLAGS

def main(_):

    with tf.Session() as sess:
        mnist = MNIST(sess, input_width=FLAGS.input_width, input_height=FLAGS.input_height, checkpoint_dir=FLAGS.checkpoint_dir)
        
        if FLAGS.is_train:
            mnist.trainmodel(FLAGS)
        else:
            if not mnist.load(FLAGS.checkpoint_dir):
                raise Exception("[!] Train a model first, then run test mode")

        mnist.get_accuracy()

if __name__ == '__main__':
    tf.app.run()
