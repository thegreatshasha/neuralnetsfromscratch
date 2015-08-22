import numpy as np

# Some common helper methods sigmoid, sigmoid_vec, sigmoid_prime, sigmoid_prime_vec
def sigmoid(z):
	return 1/(1+np.exp(-z))

def sigmoid_prime(z):
	a = sigmoid(z)
	return a * (1-a)

def exp(z):
	return np.exp(z)

def softmax_vec(z):
	num = exp_vec(z)
	denom = sum(num)
	return num/denom

sigmoid_vec = np.vectorize(sigmoid)
sigmoid_prime_vec = np.vectorize(sigmoid_prime)
exp_vec = np.vectorize(exp)