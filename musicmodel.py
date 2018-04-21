from __future__ import print_function, with_statement

import tensorflow as tf
from sklearn.utils import shuffle
import pdb

from data.convertmidi import *

print("Running tf version {}".format(tf.__version__))

# Hyperparameters
TIME_STEPS = 60
N_FEATURES = 128
N_EMBED = 64
N_HIDDEN = 256
N_OUTPUT = N_FEATURES
N_EPOCHS = 100
BATCH_SIZE = 30
ETA = .01
n_lstm_layers = 2
keep_prob = 0.5

N_SEED = 60

def f(X):
    """
    Custom non-linear activation function for MIDI.
    Ensures velocity is a valid value between 0 and 128
    """
    return tf.minimum(
        tf.maximum(X, 0),
        128
    )

def stats(arr):
	sparsity = (np.sum(np.count_nonzero(arr)).astype(np.float)) / np.size(arr)

	print("  shape: {}".format(arr.shape))
	print("  sparsity (non-zero elements): %{}".format(sparsity * 100))
	print("  values range from {} to {}".format(np.amin(arr), np.amax(arr)))

class MusicGen:

    def __init__(self):

        # TODO parametrize constants as args here

        # LSTM cell
        self.stacked_lstm = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN)
        # lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob) # dropout between lstm layers
        # self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_dropout, tf.contrib.rnn.BasicLSTMCell(N_HIDDEN)])

        # Output layer parameters
        self.W = tf.Variable(tf.random_normal([N_HIDDEN, N_OUTPUT]))
        self.b = tf.Variable(tf.zeros([N_OUTPUT]))

    def add_train_graph(self):

        with tf.name_scope("train"):

            print("Building train graph..")

            # Input tensor
            X = tf.placeholder(tf.float32, shape=[BATCH_SIZE, TIME_STEPS, N_FEATURES])

            hidden_state = tf.zeros(shape=[BATCH_SIZE, N_HIDDEN])
            current_state = tf.zeros([BATCH_SIZE, N_HIDDEN])
            state = hidden_state, current_state

            predictions = []
            loss = 0.

            for t, x_t in enumerate(tf.unstack(X, axis=1)):
                if t == X.shape[1] - 1: # No prediction for last thing
                   continue

                h, state = self.stacked_lstm(x_t, state)
                y = f(tf.matmul(h, self.W) + self.b) # piano roll prediction
                # y = tf.matmul(h, self.W) + self.b # piano roll prediction


                predictions.append(y)

                loss += tf.losses.mean_squared_error(X[:,t + 1,:], y)
                loss += tf.cast(tf.count_nonzero(tf.cast(y, tf.int64)), tf.float32) # TODO make this smooth

                # TODO I think we might want to use something besides MSE? Should check papers

            predictions = tf.cast(tf.stack(predictions, axis=1), tf.int64)
            train = tf.train.AdamOptimizer(ETA).minimize(tf.reduce_sum(loss))

        print("Graph built successfully!")
        return X, predictions, loss, train

    def add_gen_graph(self):

        with tf.name_scope("gen"):

            self.x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_FEATURES])
            self.hidden_state = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_HIDDEN])
            self.current_state = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_HIDDEN])
            state = self.hidden_state, self.current_state

            # TODO link this to code above
            h, state = self.stacked_lstm(self.x, state)
            self.y = f(tf.matmul(h, self.W) + self.b)
            # self.y = tf.matmul(h, self.W) + self.b

            self.next_hidden_state, self.next_current_state = state

    def train(self, X, session):
        for epoch in xrange(N_EPOCHS):
            X = shuffle(X)
            loss = 0.
            for i in xrange(0, len(X) - BATCH_SIZE, BATCH_SIZE):
                batch_X = X[i:i + BATCH_SIZE, :, :]
                batch_loss, _ = session.run([loss_op, train_op], feed_dict={
                    X_placeholder: batch_X,
                })
                loss += batch_loss
            print("Epoch {} train loss: {}".format(epoch, loss))

    # TODO this function will not "generate" songs.. but that won't be too different
    def predict(self, X, session, length=3600):

        hidden_state = np.zeros(shape=[BATCH_SIZE, N_HIDDEN])
        current_state = np.zeros(shape=[BATCH_SIZE, N_HIDDEN])

        for i in xrange(X.shape[1]):
            y, hidden_state, current_state = session.run([self.y, self.next_hidden_state, self.next_current_state], feed_dict={
                self.x: X[:,i,:],
                self.hidden_state: hidden_state,
                self.current_state: current_state,
            })

        predictions = []
        for i in xrange(length):
            y, hidden_state, current_state = session.run([self.y, self.next_hidden_state, self.next_current_state], feed_dict={
                self.x: y,
                self.hidden_state: hidden_state,
                self.current_state: current_state,
            })
            predictions.append(y)

        return np.stack(predictions, axis=1).astype(np.int64)

if __name__ == "__main__":

    print("Loading data..")
    data_o, attributes = midi_encode("data/songs/moonlightinvermont.mid", False)
    data = np.array(data_o)
    data = data[:len(data) - (len(data) % TIME_STEPS),:] # Cut off extra stuff
    stats(data)
    data = np.stack(np.split(data, TIME_STEPS, axis=0), axis=1) # Cut song into separate samples of same length
    # The data should have shape (num_trials, time_steps, num_features)
    print("Training data of shape {}".format(data.shape))

    gen = MusicGen()

    X_placeholder, predict_op, loss_op, train_op = gen.add_train_graph()

    session = tf.Session()
    print("Initializing all variables")
    session.run(tf.global_variables_initializer())

    print("Training..")
    gen.train(data, session)
    print("Training completed!")

    print("Predicting..")
    gen.add_gen_graph()
    predictions = gen.predict(data[:BATCH_SIZE,:N_SEED, :], session)
    print("  prediction tensor of shape {}".format(predictions.shape))
    print("Encoding MIDI..")
    prediction = predictions[0,:,:]
    stats(prediction)
    prediction = prediction.tolist()

    pattern = midi_decode(prediction, attributes)
    midi.write_midifile("output.mid", pattern)

    print("Got prediction tensor of shape {}".format(predictions.shape))
