import numpy as np

# Some common helper methods sigmoid, sigmoid_vec, sigmoid_prime, sigmoid_prime_vec
def sigmoid(z):
	return 1/(1+np.exp(-z))

def sigmoid_prime(z):
	a = sigmoid(z)
	return a * (1-a)

sigmoid_vec = np.vectorize(sigmoid)
sigmoid_prime_vec = np.vectorize(sigmoid_prime)