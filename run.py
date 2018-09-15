import numpy as np

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z)
	return z * (1 - z)

# todo try on old examples(with predicting by first value)

# neural network with 3 input units
class NeuralNetwork:
	def __init__(self, x, y):
		self.input = x
		# todo resolve in future
		# self.weights1   = np.random.rand(self.input.shape[1],4) 
		# todo add biases
		self.weights_1 = np.random.rand(4, 4)
		self.weights_2 = np.random.rand(1, 5)
		self.y = y
		self.output = np.zeros(y.shape)


	def forward_propagate(self):
		input_rows_count = len(self.input_with_biases)
		ones = np.ones((input_rows_count,1), float)
		input_with_biases = np.concatenate((ones, a), axis=1)
		for row in range(input_rows_count):
			# activation for 1-st layer it's just input row
			# with added bias unit
			activation_1 = input_with_biases[row, :]
			# activation for 2-nd layer
			z_2 = np.dot(self.weights_1, activation_1.T) # (4, 1)
			activation_2 = sigmoid(z)
			# adding bias
			activation_2 = np.concatenate((np.ones((1,1), float), activation_2), axis=0) # (5, 1)
			# output in 3-rd layer
			z_3 = np.dot(self.weights_2, activation_2)
			output = sigmoid(z)
			self.output[row] = output

	def back_propagate(self)
	





