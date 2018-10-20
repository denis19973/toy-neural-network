import numpy as np
from random import shuffle


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
	return x * (1 - x)


class NeuralNetwork:
	LEARNING_RATE = .5
	CROSS_VALIDATION_PERCENT = .35
	layers = []
	dataset = []

	def __init__(self, dataset, n_inputs, n_outputs, n_hidden=4):
		self.dataset = dataset
		self.n_outputs = n_outputs
		hidden_layer = [{'weights': [np.random.rand() for _ in range(n_inputs + 1)]} for _ in range(n_hidden)]
		self.layers.append(hidden_layer)
		output_layer = [{'weights': [np.random.rand() for _ in range(n_hidden + 1)]} for _ in range(n_outputs)]
		self.layers.append(output_layer)

	def forward_propagate(self, row):
		inputs = row
		for layer in self.layers:
			new_inputs = []
			for neuron in layer:
				activation = self.activate(neuron['weights'], inputs)
				neuron['output'] = sigmoid(activation)
				new_inputs.append(neuron['output'])
			inputs = new_inputs
		return inputs

	def activate(self, weights, inputs):
		"""Calculate activation for single neuron"""
		activation = weights[-1]  # weights[-1] it's bias
		for i in range(len(weights) - 1):
			activation += weights[i] * inputs[i]
		return activation

	def backward_propagate_error(self, expected):
		"""Backpropagate error and store in neurons"""
		for i in reversed(range(len(self.layers))):
			layer = self.layers[i]
			errors = []
			if i != len(self.layers) - 1:
				for j in range(len(layer)):
					error = 0
					for neuron in self.layers[i + 1]:
						error += neuron['weights'][j] * neuron['delta']
					errors.append(error)
			else:
				for j in range(len(layer)):
					neuron = layer[j]
					errors.append(expected[j] - neuron['output'])
			for j in range(len(layer)):
				neuron = layer[j]
				neuron['delta'] = errors[j] * sigmoid_deriv(neuron['output'])

	def update_weights(self, row, l_rate):
		for i in range(len(self.layers)):
			inputs = row[:-1]
			if i != 0:
				inputs = [neuron['output'] for neuron in self.layers[i - 1]]
			for neuron in self.layers[i]:
				for j in range(len(inputs)):
					neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
				neuron['weights'][-1] += l_rate * neuron['delta']

	def train(self, n_epoch, l_rate=None):
		l_rate = l_rate or self.LEARNING_RATE
		cross_validation_index = int(len(self.dataset) * self.CROSS_VALIDATION_PERCENT)
		shuffle(self.dataset)
		cv_dataset = self.dataset[:cross_validation_index]
		dataset = self.dataset[cross_validation_index:]
		for epoch in range(n_epoch):
			sum_error = 0
			for row in dataset:
				outputs = self.forward_propagate(row)
				if self.n_outputs > 1:
					expected = [0 for i in range(self.n_outputs)]
					expected[row[-1] - 1] = 1
				else:
					expected = [row[-1]]
				sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
				self.backward_propagate_error(expected)
				self.update_weights(row, l_rate)
			print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
		print('Cross validation error={}'.format(self.get_cross_validation_error(cv_dataset)))

	def predict(self, row):
		outputs = self.forward_propagate(row)
		if len(outputs) > 1:
			return outputs.index(max(outputs)) + 1
		else:
			return outputs[0]

	def get_cross_validation_error(self, cv_dataset, threshold=0.5):
		sum_error = 0
		for cv_row in cv_dataset:
			predicted = self.predict(cv_row)
			expected_class = cv_row[len(cv_row) - 1]
			if predicted != expected_class:
				sum_error += 1
		return sum_error / len(cv_dataset)
