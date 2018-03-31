""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from six.moves import cPickle as pickle
import sys
import math
import time

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

DATA_PATH = 'medical_data/'
DATA_FILE = DATA_PATH + 'medical_data.pickle'
INCLUDE_TEST_SET = False


print("Loading datasets...")
with open(DATA_FILE, 'rb') as f:
  save = pickle.load(f)
  train_X = save['train_data']
  train_Y = save['train_labels']
  val_X = save['val_data']
  val_Y = save['val_labels']

  if INCLUDE_TEST_SET:
    test_X = save['test_data']
    test_Y = save['test_labels']
  del save  # hint to help gc free up memory

print('Training set', train_X.shape, train_Y.shape)
print('Validation set', val_X.shape, val_Y.shape)
if INCLUDE_TEST_SET:
  print('Test set', test_X.shape, test_Y.shape)



'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
learning_rate = 0.01
training_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
num_input = 50 # MNIST data input (img shape: 28*28)
timesteps = 50 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 15 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    #lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    lstm_cell = rnn.AttentionCellWrapper(
                cell=rnn.BasicLSTMCell(num_hidden, forget_bias=1.0),
                attn_length= 10)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Calculate accuracy for 128 mnist test images
val_len = val_Y.shape[0]
valid_x = val_X.reshape((-1, timesteps, num_input))
valid_y = val_Y

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        offset = (step * batch_size) % (train_Y.shape[0] - batch_size)
        batch_data = train_X[offset:(offset + batch_size), :, :, :]
        batch_y = train_Y[offset:(offset +batch_size), :]
        batch_x = batch_data.reshape((batch_size, timesteps, num_input))

        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        #batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            print("Validation Accuracy:", \
            sess.run(accuracy, feed_dict={X: valid_x, Y: valid_y}))

    print("Optimization Finished!")

    
    
    """
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

    """