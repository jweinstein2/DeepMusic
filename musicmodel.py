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
N_HIDDEN = 64
N_OUTPUT = N_FEATURES
N_EPOCHS = 10
BATCH_SIZE = 10
ETA = .01
n_lstm_layers = 2

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

training = tf.placeholder_with_default(False, shape=())
keep_prob = tf.cond(training, lambda:tf.constant(0.5), lambda:tf.constant(1.0))

def lstm_dropout_cell():
	lstm = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN)
	lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob) # dropout between lstm layers
	return lstm_dropout

class MusicGen:

	# def __init__(self):

	#   # TODO parametrize constants as args here
		

	def add_train_graph(self):
		with tf.name_scope("train"):
			print("Building tf graph..")

			# Input tensor
			X = tf.placeholder(tf.float32, shape=[BATCH_SIZE, TIME_STEPS, N_FEATURES])

			# TODO maybe add an embedding layer here?

			# LSTM cell
			
			self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_dropout_cell() for _ in range(n_lstm_layers)])

			predictions = []
			loss = 0.

			initial_state = self.stacked_lstm.zero_state(BATCH_SIZE, dtype=tf.float32)

			outputs, state = tf.nn.dynamic_rnn(self.stacked_lstm, X,
										   initial_state=initial_state,
										   dtype=tf.float32)


			predictions = tf.layers.dense(outputs, N_OUTPUT, activation=f, name='fc')

			loss += tf.losses.mean_squared_error(X, predictions)

			predictions = tf.cast(predictions, tf.int64)

			# predictions = tf.cast(tf.stack(predictions, axis=1), tf.int64)
			train = tf.train.AdamOptimizer(ETA).minimize(loss)

		print("Graph built successfully!")
		return X, predictions, loss, train

	def add_gen_graph(self):

		with tf.name_scope("gen"):

			self.x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1, N_FEATURES]) #1 time step

			self.hidden_state1 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_HIDDEN])
			self.current_state1 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_HIDDEN])
			self.hidden_state2 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_HIDDEN])
			self.current_state2 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_HIDDEN])

			state1 = tf.contrib.rnn.LSTMStateTuple(self.hidden_state1, self.current_state1)
			state2 = tf.contrib.rnn.LSTMStateTuple(self.hidden_state2, self.current_state2)

			state = (state1, state2)

			outputs, state = tf.nn.dynamic_rnn(self.stacked_lstm, self.x,
										   initial_state=state,
										   dtype=tf.float32)


			(self.next_hidden_state1, self.next_current_state1), (self.next_hidden_state2, self.next_current_state2) = state

			self.y = tf.layers.dense(outputs, N_OUTPUT, activation=f, name='fc', reuse=True) #use same weights as in train

	def train(self, X, session):
		for epoch in xrange(N_EPOCHS):
			X = shuffle(X)
			loss = 0.
			for i in xrange(0, len(X) - BATCH_SIZE, BATCH_SIZE):
				batch_X = X[i:i + BATCH_SIZE, :, :]
				batch_loss, _ = session.run([loss_op, train_op], feed_dict={
					X_placeholder: batch_X,
					training: True
				})
				loss += batch_loss
			print("Epoch {} train loss: {}".format(epoch, loss))
			# if epoch % 5 == 0:
			#     saver.save(session, './model/' + 'model.ckpt', global_step=epoch+1)
			#     print('Saved at epoch' + str(epoch))

	# TODO this function will not "generate" songs.. but that won't be too different
	def predict(self, X, session, length=3600):
		hidden_state1 = np.zeros(shape=[BATCH_SIZE, N_HIDDEN])
		current_state1 = np.zeros(shape=[BATCH_SIZE, N_HIDDEN])
		hidden_state2 = np.zeros(shape=[BATCH_SIZE, N_HIDDEN])
		current_state2 = np.zeros(shape=[BATCH_SIZE, N_HIDDEN])

		for i in xrange(X.shape[1]):
			y, hidden_state1, current_state1, hidden_state2, current_state2 = session.run(
				[self.y, self.next_hidden_state1, self.next_current_state1, self.next_hidden_state2, self.next_current_state2], 
				feed_dict={
					self.x: X[:,i:i+1,:],
					self.hidden_state1: hidden_state1,
					self.current_state1: current_state1,
					self.hidden_state2: hidden_state2,
					self.current_state2: current_state2
			})

		predictions = []
		for i in xrange(length):
			y, hidden_state1, current_state1, hidden_state2, current_state2 = session.run(
				[self.y, self.next_hidden_state1, self.next_current_state1, self.next_hidden_state2, self.next_current_state2], 
				feed_dict={
					self.x: y,
					self.hidden_state1: hidden_state1,
					self.current_state1: current_state1,
					self.hidden_state2: hidden_state2,
					self.current_state2: current_state2
			})
			predictions.append(y)

		return np.stack(predictions, axis=1).astype(np.int64) 

if __name__ == "__main__":

	print("Loading data..")
	data_o = midi_encode("data/songs/moonlightinvermont.mid")
	data = np.array(data_o)
	data = data[:len(data) - (len(data) % TIME_STEPS),:] # Cut off extra stuff
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
	prediction = predictions[0,:,0,:]
	print("  values range from {} to {}".format(np.amin(prediction), np.amax(prediction)))
	prediction = prediction.tolist()

	pattern = midi_decode(prediction)
	print(pattern)
	midi.write_midifile("testoutput.mid", pattern)

	print("Got prediction tensor of shape {}".format(predictions.shape))
