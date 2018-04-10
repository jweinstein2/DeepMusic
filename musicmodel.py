from __future__ import print_function

import tensorflow as tf

print("Running tf version {}".format(tf.__version__))

# Parameters and hyperparameters
TIME_STEPS = 60
BATCH_SIZE = 10
N_FEATURES = 128
N_HIDDEN = 64
N_OUTPUT = 128

# Input tensor
X = tf.placeholder(tf.float32, shape=[TIME_STEPS, BATCH_SIZE, N_FEATURES])

# LSTM cell
lstm = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN)

# Softmax layer parameters
W = tf.Variable(tf.random_normal([N_HIDDEN, N_OUTPUT]))
b = tf.Variable(tf.zeros([N_OUTPUT]))

hidden_state = tf.zeros(shape=[BATCH_SIZE, N_HIDDEN])
current_state = tf.zeros([BATCH_SIZE, N_HIDDEN])
state = hidden_state, current_state

predictions = []
loss = 0.

for x_t in tf.unstack(X, axis=0):
    h, state = lstm(x_t, state)
    y = tf.matmul(h, W) + b
    predictions.append(tf.nn.softmax(y))
    loss += 0 # TODO compute some loss function
