# Main driver program
import numpy as np
from mnist_loader import load_data_wrapper
from neural import Neural

training_data, validation_data, test_data = load_data_wrapper()

nn = Neural([784, 50, 10])
input = np.ones((784, 1))
nn.feedforward(input)
nn.SGD(training_data, test_data, 3, 25, 50)
import pdb; pdb.set_trace()
