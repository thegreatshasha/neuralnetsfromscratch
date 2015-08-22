#Implement a simple neural network first which works with mnist data and predicts images

#Start off with logistic units
import numpy as np
from helpers import sigmoid_vec, sigmoid_prime_vec, softmax_vec
from random import shuffle

#feedforward, SGD, update_mini_batch, backwardspropogation

#loop form of fundamnetal eqns of backpropogation

class Neural:
	def __init__(self, layers, vocab):
		self.layers = np.array(layers)
		self.word_length, self.num_words = layers[0]
		self.vocab = vocab
		
		# Initialize weights
		self.weights = [np.random.randn(self.num_words, self.layers[1], self.word_length)]
		 
		for index in xrange(1, len(layers)-1): #l-1 weights
			val = layers[index]
			self.weights.append(np.random.randn(layers[index+1], val))
		# weights[0] is a [layers[1] matrix*m]

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
		

	def feedforward(self, input):
		# input is an array of word indexes like [1,250, 400]

		# three step feedforward calculation
		activation = np.zeros(self.biases[0].shape)
		# first layer, for each word in input, lookup corresponding vector and calculate activations
		for index, val in enumerate(input):
			word_vec = self.word_vector(index)
			word_vec_weights = self.weights[0][index]
			#import pdb; pdb.set_trace()
			activation = activation + np.dot(word_vec_weights, word_vec)

		activation = sigmoid_vec(activation)
		activations = [activation]
		
		# Middle layers calculate logistic normally
		for index, value in enumerate(zip(self.weights[1:-1], self.biases[1:-1])):
			weight, bias = value
			activation = sigmoid_vec(np.dot(weight, activation) + bias)
			activations.append(activation)
		
		# Last layer, do sigmoid_vec calculation and return
		activation = softmax_vec(np.dot(self.weights[-1], activation))
		activations.append(activation)
		
		#import pdb; pdb.set_trace()

		return activations

	def SGD(self, training_data, test_data, eta, epochs, batch_size):
		print "Split input into multiple batches of constant batch size"
		print "Randomly Shuffle in each epoch"

		# Repeat process over multiple epochs
		for epoch in xrange(0, epochs):
			shuffle(training_data)

			for i in xrange(0, len(training_data), batch_size):
				#print "Running batch {}/{}".format(i/batch_size												, len(training_data)/batch_size)

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

		for data in batch:
			deltas, deltas_weights = self.backpropogate(data[0], data[1])
			# iterate and average deltas
			for i, delta in enumerate(deltas_avg):
				deltas_avg[i] = deltas_avg[i] + deltas[i]/batch_size

				#iterate and average 
				deltas_weights_avg[i] = deltas_weights_avg[i] + deltas_weights[i]/batch_size

		# Change internal weights and biases with this averaged data
		for i, delta in enumerate(deltas_avg):
			#import pdb; pdb.set_trace()
			self.weights[i] = self.weights[i] - eta * deltas_weights_avg[i]
			self.biases[i] = self.biases[i] - eta * deltas_avg[i] #deltas are same as bias

	def backpropogate(self, x, y):
		#print "Do one feedforward pass and calculate errors"
		#print "Backpropogate these errors and update weights and biases"
		activations = self.feedforward(x)
		deltas = [0] * (len(self.layers) - 1)
		deltas_weights = [0] * (len(self.layers) - 1)
		# No need for delta_biases since it's equivalent to deltas

		# Do delta calculation for last layer

		# Multiply gradient with sigmoid derivative to calculate error
		error = self.gradient(activations[-1], y) * (activations[-1]*(1-activations[-1]))

		deltas[-1] = error
		deltas_weights[-1] = np.dot(deltas[-1], np.transpose(activations[-2]))

		# Go back layer by layer and calculate delta for middle layers
		for index in xrange(1, len(self.layers) - 1):
			#print index, deltas[-index-1]
			deltas[-index-1] = np.dot(np.transpose(self.weights[-index]), deltas[-index]) * (activations[-index-1]*(1-activations[-index-1]))
			deltas_weights[-index-1] = np.dot(deltas[-index-1], np.transpose(activations[-index-2]))

		# Do delta calculations for first layer

		return (deltas, deltas_weights)


	def evaluate(self, test_data):
		print "Evaluate number of positive examples/total examples"
		count = 0

		for i, data in enumerate(test_data):
			#import pdb; pdb.set_trace()
			activations = self.feedforward(data[0])
			#import pdb; pdb.set_trace()
			result = (np.argmax(activations[-1]) == data[1])
			if(result == True):
				count += 1
				#print "{}/{}: {}".format(i, len(test_data), data[1])

		print "Success {}/{}".format(count, len(test_data))