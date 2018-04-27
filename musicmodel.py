from __future__ import print_function, with_statement, division

import tensorflow as tf
from sklearn.utils import shuffle
import random, argparse
import pdb

from data.convertmidi import *
from data.convert import *

print("Running tf version {}".format(tf.__version__))

# Hyperparameters
TIME_STEPS = 120
N_FEATURES = 128
N_EMBED = 1028
N_HIDDEN = 1028
# N_OUTPUT = # DECIDED BY OH_FEATURES
N_EPOCHS = 200 # Seem to need to train for 100 to get anything
BATCH_SIZE = 10
ETA = .01
n_lstm_layers = 2
keep_prob = 0.5

EPSILON = 0.5

START = 5
N_SEED = 120

def f(X):
    """
    Custom non-linear activation function for MIDI.
    Ensures velocity is a valid value between 0 and 128
    """
    return tf.minimum(
        tf.maximum(X, 0),
        128
    )

def sample_bernoulli(p):
    return np.apply_along_axis(
        lambda x: random.random() < x,
        axis=1,
        arr=p,
    ).astype(np.int32)

def sample(a, temperature=0.1):
    # a = np.log(a) / temperature
    # dist = np.exp(a)/np.sum(np.exp(a))
    choices = range(len(a))
    dist = np.array(a)**(1/temperature)
    dist /= sum(dist)
    index = np.random.choice(choices, p=dist)

    selected = np.zeros(a.shape[0])
    selected[index] = 1
    return selected

def stats(pred_hold, pred_hit=None):
    print("Stats:")
    print("  held shape:", pred_hold.shape)
    # print("  hit shape:", pred_hit.shape)
    print("  mean notes held / t:", np.mean(np.sum(pred_hold, axis=1)))
    # print("  mean notes hit / t:", np.mean(np.sum(pred_hit, axis=1)))
    print("  mean notes held:", np.mean(pred_hold))
    # print("  mean notes hit:", np.mean(pred_hit))
    print("  sustained:", np.mean(np.sum(np.multiply(np.roll(pred_hold, 1, axis=0), pred_hold), axis=1)))

class MusicGen:

    def __init__(self):

        # TODO parametrize constants as args here

        # LSTM cell
        self.stacked_lstm = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN)
        # lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob) # dropout between lstm layers
        # self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_dropout, tf.contrib.rnn.BasicLSTMCell(N_HIDDEN)])

        # Embedding layer parameters
        self.We = tf.get_variable("We", shape=[N_FEATURES * 2, N_EMBED], initializer=tf.contrib.layers.xavier_initializer())
        self.be = tf.get_variable("be", shape=[N_EMBED], initializer=tf.zeros_initializer())

        # Output layer parameters
        self.W = tf.get_variable("W", shape=[N_HIDDEN, N_OUTPUT * 2], initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable("b", shape=[N_OUTPUT * 2], initializer=tf.zeros_initializer())

    def add_train_graph(self):

        print("Building train graph..")

        with tf.name_scope("train"):

            # Input tensors
            self.X_hold = tf.placeholder(tf.float32, shape=[BATCH_SIZE, TIME_STEPS, N_FEATURES])
            self.X_hold_len = tf.placeholder(tf.float32, shape=[BATCH_SIZE, TIME_STEPS, N_FEATURES])
            self.Y_hold = tf.placeholder(tf.float32, shape=[BATCH_SIZE, TIME_STEPS, N_OUTPUT])

            X = tf.concat([self.X_hold, self.X_hold_len], axis=2)

            hidden_state = tf.zeros(shape=[BATCH_SIZE, N_HIDDEN])
            current_state = tf.zeros([BATCH_SIZE, N_HIDDEN])
            state = hidden_state, current_state

            self.loss = 0.
            self.perp = 0.

            for t, x_t in enumerate(tf.unstack(X, axis=1)):
                if t == X.shape[1] - 1: # No prediction for last thing
                    continue

                # Embedding layer
                e = tf.nn.relu(tf.matmul(x_t, self.We) + self.be)

                # Recurrent layer
                h, state = self.stacked_lstm(e, state)

                # Output layer
                y = tf.matmul(h, self.W) + self.b
                y_hold, _ = tf.split(y, [N_OUTPUT, N_OUTPUT], axis=1)

                # multiclass sigmoid
                # self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hold, labels=self.X_hold[:,t + 1]))

                # softmax over tokens (deepjazz)
                self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hold, labels=self.Y_hold[:,t + 1]))

                # calculate perplexity per note
                p = tf.nn.softmax(y_hold)
                self.perp += tf.reduce_mean(tf.pow(2., -tf.reduce_sum(p * tf.log(p), axis=1)))

            self.perp /= t + 1
            self.loss /= t + 1
            self.train_step = tf.train.AdamOptimizer(ETA).minimize(tf.reduce_sum(self.loss))

        print("Train graph built successfully!")

    def add_gen_graph(self):

        print("Building gen graph..")

        with tf.name_scope("gen"):

            self.x_hold = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_FEATURES])
            self.x_hold_len = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_FEATURES])
            x = tf.concat([self.x_hold, self.x_hold_len], axis=1)

            e = tf.nn.relu(tf.matmul(x, self.We) + self.be)

            self.hidden_state = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_HIDDEN])
            self.current_state = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_HIDDEN])
            state = self.hidden_state, self.current_state
            h, state = self.stacked_lstm(e, state)

            y = tf.matmul(h, self.W) + self.b
            self.y_hold, self.y_hold_len = tf.split(y, [N_OUTPUT, N_OUTPUT], axis=1)
            self.y_hold = tf.nn.softmax(self.y_hold, axis=1) #sigmoid
            # self.y_hit = tf.nn.softmax(self.y_hit, axis=1)

            self.next_hidden_state, self.next_current_state = state

        print("Gen graph build successfully!")

    def train(self, X_hold, X_hold_len, session):
        for epoch in xrange(N_EPOCHS):
            X_hold, X_hit = shuffle(X_hold, X_hold_len)
            min_loss = loss = 0.
            perp = 0.
            epochs_wo_improvement = 0

            for i in xrange(0, len(X_hold) - BATCH_SIZE, BATCH_SIZE):
                batch_perp, batch_loss, _ = session.run([self.perp, self.loss, self.train_step], feed_dict={
                    self.X_hold: X_hold[i:i + BATCH_SIZE, :, :],
                    self.X_hold_len: X_hold_len[i:i + BATCH_SIZE, :, :],
                    self.Y_hold: Y_hold[i:i + BATCH_SIZE, :, :],
                    # self.Y_hold: Y_hold[i:i + BATCH_SIZE, :],
                    # self.Y_hit: Y_hit[i:i + BATCH_SIZE, :],
                })
                perp += batch_perp
                loss += batch_loss
            den = i + 1
            print("Epoch {} train loss/perp: {}, {}".format(epoch, loss / den, perp / den))

            # early stopping
            if (loss < min_loss):
                epochs_wo_improvent = 0
                min_loss = loss
            else: epochs_wo_improvement += 1
            if epochs_wo_improvement > 16:
                print("Early stopping...")
                break

    def predict(self, x_hold, x_hold_len, session, le, length=3600):

        hidden_state = np.zeros(shape=[BATCH_SIZE, N_HIDDEN])
        current_state = np.zeros(shape=[BATCH_SIZE, N_HIDDEN])

        nodes = [self.y_hold, self.y_hold_len, self.next_hidden_state, self.next_current_state]

        for i in xrange(x_hold.shape[1]):
            y_hold, _, hidden_state, current_state = session.run(nodes, feed_dict={
                self.x_hold: x_hold[:,i,:],
                self.x_hold_len: x_hold_len[:,i,:],
                self.hidden_state: hidden_state,
                self.current_state: current_state,
            })

        # TODO Several options for generation:
        #   * Round each probability
        #   * Sample stochastically according to distribution
        #   * Feed in probability distribution at next prediction

        pred_hold = []
        cur_hold_len = x_hold[:,x_hold.shape[1] - 1,:]
        for i in xrange(length):

            # y_hold = (y_hold > EPSILON).astype(np.int32)
            # y_hit = (y_hit > EPSILON).astype(np.int32)
            # calculate next x_hold

            # update hold_len
            y_hold = np.apply_along_axis(sample, 1, y_hold)
            pred_hold.append(y_hold)
            y_hold = multihot(y_hold, le)
            cur_hold_len = (cur_hold_len + y_hold) * y_hold

            y_hold, _, hidden_state, current_state = session.run(nodes, feed_dict={
                self.x_hold: y_hold,
                self.x_hold_len: cur_hold_len,
                self.hidden_state: hidden_state,
                self.current_state: current_state,
            })

        return np.stack(pred_hold, axis=1)

crop_data = lambda data: data[:len(data) - (len(data) % TIME_STEPS),:]
stack = lambda data: np.stack(np.split(data, TIME_STEPS, axis=0), axis=1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train and generate music model.')
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()
    args.train = not args.load

    # print("Loading data..")
    # data_o, attributes = midi_encode("data/songs/moonlightinvermont.mid", False)
    # data = np.array(data_o)
    # data = data[:len(data) - (len(data) % TIME_STEPS),:] # Cut off extra stuff
    # stats(data)
    # data =  # Cut song into separate samples of same length
    # # The data should have shape (num_trials, time_steps, num_features)
    # print("Training data of shape {}".format(data.shape))

    print("Loading data..")
    raw_hold, raw_hit, raw_hold_len, attr = encode("data/songs/moonlightinvermont.mid", False)
    raw_hold, _, raw_hold_len = map(crop_data, [raw_hold, raw_hit, raw_hold_len])
    oh_hold, le = onehot(raw_hold)
    stats(raw_hold)
    N_OUTPUT = len(le.classes_)
    print("num features: {}".format(N_OUTPUT))

    # Stack by timesteps
    X_hold, X_hold_len, Y_hold = map(stack, [raw_hold, raw_hold_len, oh_hold])

    print("Calculating seed..")
    seed_hold = X_hold[START:START + BATCH_SIZE,:N_SEED, :]
    seed_hold_len = X_hold_len[START:START + BATCH_SIZE,:N_SEED, :]
    # stats(seed_hold[0,:,:], seed_hit[0,:,:])

    gen = MusicGen()
    # gen.add_train_graph()
    gen.add_gen_graph()

    saver = tf.train.Saver()

    session = tf.Session()
    print("Initializing all variables")
    session.run(tf.global_variables_initializer())

    if args.train:
        print("Training..")
        gen.train(X_hold, X_hit, session)
        saver.save(session, "models/recent")
        print("Training completed!")

    else:
        print("Restoring..")
        saver.restore(session, "models/recent")
        print("Model models/recent restored!")

    print("Predicting..")
    pred_hold = gen.predict(seed_hold, seed_hold_len, session, le)
    print("Predicted tensors of shapes {}".format(pred_hold.shape))

    print("Encoding MIDI..")
    pred_hold = pred_hold[0,:,:]
    stats(pred_hold)
    mh_hold = multihot(pred_hold, le)
    pattern = decode(mh_hold, None, attr)
    midi.write_midifile("output.mid", pattern)
    print("MIDI saved!")
