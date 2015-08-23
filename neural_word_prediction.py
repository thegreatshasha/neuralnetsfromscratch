#Implement a simple neural network first which works with mnist data and predicts images

#Start off with logistic units
import numpy as np
from helpers import sigmoid_vec, sigmoid_prime_vec, softmax_vec
from random import shuffle

#feedforward, SGD, update_mini_batch, backwardspropogation

#loop form of fundamnetal eqns of backpropogation

class Neural:
	def __init__(self, layers, vocab, num_words):
		self.layers = np.array(layers)
		self.vocab = vocab
		self.num_words = num_words
		self.word_length = self.layers[0]/self.num_words
		
		# Initialize weights
		self.weights = [np.random.randn(layers[index+1], val) for index,val in enumerate(layers[:-1])]
		 
		# Initialize biases
		self.biases = [np.random.randn(val, 1) for val in layers[1:]]
		self.lookup_matrix = np.random.randn(len(self.vocab), self.word_length, 1)
		print "It is not very clear now either"

	def word_vector(self, index):
		# this returns vector of activations for a word vector from index
		return self.lookup_matrix[index-1]

	def lookup_word(self, index):
		return self.vocab[index-1]

	def cross_entropy_cost(self, x, y):
		print "Returns cross entropy cost"

	def indexes_to_vector(self, indexes):
		vectors = [self.word_vector(index) for index in indexes]
		return np.concatenate(vectors)

	def index_to_softmax_vector(self, index):
		vector = np.zeros((250, 1))
		vector[index-1] = 1
		return vector

	def update_lookup_matrix(self, indexes, delta_input, batch_size, eta):
		for i, index in enumerate(indexes):
			self.lookup_matrix[index-1] = self.lookup_matrix[index-1] - eta*delta_input[i:i+self.word_length]/batch_size


	def feedforward(self, input):
		# input is an array of word indexes like [1,250, 400]. These are joined to generate a big vector

		activations = []
		activation = self.indexes_to_vector(input)
		activations.append(activation)
		
		for index, value in enumerate(zip(self.weights, self.biases)):
			#import pdb; pdb.set_trace()
			weight, bias = value
			nonlinear_method = sigmoid_vec if (index < len(self.biases) -1) else softmax_vec
			activation = nonlinear_method(np.dot(weight, activation) + bias)
			activations.append(activation)
		#print "I will propogate this input and return the activations"
		return activations

	def SGD(self, training_data, test_data, eta, epochs, batch_size):
		print "Split input into multiple batches of constant batch size"
		print "Randomly Shuffle in each epoch"

		# Repeat process over multiple epochs
		for epoch in xrange(0, epochs):
			shuffle(training_data)

			for i in xrange(0, len(training_data), batch_size):
				print "Running batch {}/{}".format(i/batch_size												, len(training_data)/batch_size)

				batch = training_data[i:i+batch_size]

				self.update_mini_batch(batch, batch_size, eta)

				# run any validation etc
			self.evaluate(test_data)

	def gradient(self, output, y):
		return output - y

	def update_mini_batch(self, batch, batch_size, eta):

		#print "Run backpropogation on each bach and calculate avg deltas in weights and biases and then update them"
		#variables for summing deltas
		deltas_avg = [0] * (len(self.layers) - 1)
		deltas_weights_avg = [0] * (len(self.layers) - 1)
		deltas_lookup_matrix = np.zeros(self.lookup_matrix.shape).shape

		for i, data in enumerate(batch):
			#print "Batch: {}/{}".format(i+1, batch_size)
			deltas, deltas_weights, delta_input = self.backpropogate(data[0:-1], data[-1])
			
			# update lookup table
			self.update_lookup_matrix(data[0:-1], delta_input, batch_size, eta)

			# iterate and average deltas
			for i, delta in enumerate(deltas):
				self.weights[i] = self.weights[i] - eta * deltas_weights[i]/batch_size
				self.biases[i] = self.biases[i] - eta * deltas[i]/batch_size


	def backpropogate(self, x, y):
		#print "Do one feedforward pass and calculate errors"
		#print "Backpropogate these errors and update weights and biases"
		activations = self.feedforward(x)
		deltas = [0] * (len(self.layers) - 1)
		deltas_weights = [0] * (len(self.layers) - 1)
		y = self.index_to_softmax_vector(y)		
		# Multiply gradient with sigmoid derivative to calculate error
		error = self.gradient(activations[-1], y)

		deltas[-1] = error
		deltas_weights[-1] = np.dot(deltas[-1], np.transpose(activations[-2]))

		# Go back layer by layer and calculate delta for other layers
		for index in xrange(1, len(self.layers) - 1):
			#print index, deltas[-index-1]
			deltas[-index-1] = np.dot(np.transpose(self.weights[-index]), deltas[-index]) * (activations[-index-1]*(1-activations[-index-1]))
			deltas_weights[-index-1] = np.dot(deltas[-index-1], np.transpose(activations[-index-2]))


		# Do delta calculations for first layer
		deltas_input = np.dot(np.transpose(self.weights[0]), deltas[0])

		return (deltas, deltas_weights, deltas_input)


	def evaluate(self, test_data):
		print "Evaluate number of positive examples/total examples"
		count = 0

		for i, data in enumerate(test_data):
			#import pdb; pdb.set_trace()
			activations = self.feedforward(data[:-1])
			word_index = np.argmax(activations[-1])
			#import pdb; pdb.set_trace()
			result =  (word_index+1 == data[-1])
			if(result == True):
				count += 1
				#print "{}/{}: {}".format(i, len(test_data), data[1])

		print "Success {}/{}".format(count, len(test_data))