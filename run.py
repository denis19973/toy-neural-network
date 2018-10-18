import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
	return x * (1 - x)

def initialize_network(n_inputs, n_hidden, n_outputs):
	network = []
	hidden_layer = [{'weights': [np.random.rand() for _ in range(n_inputs + 1)]} for _ in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights': [np.random.rand() for _ in range(n_hidden + 1)]} for _ in range(n_outputs)]
	network.append(output_layer)
	return network

def activate(weights, inputs):
	"""Calculate activation for single neuron"""
	activation = weights[-1]  # weights[-1] it's bias
	for i in range(len(weights) - 1):
		activation += weights[i] * inputs[i]
	return activation

def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = sigmoid(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

def backward_propagate_error(network, expected):
	"""Backpropagate error and store in neurons"""
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = []
		if i != len(network) - 1:
			for j in range(len(layer)):
				error = 0
				for neuron in network[i + 1]:
					error += neuron['weights'][j] * neuron['delta']
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * sigmoid_deriv(neuron['output'])

def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]	:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			if n_outputs > 1:
				expected = [0 for i in range(n_outputs)]
				expected[row[-1]] = 1
			else:
				expected = [row[-1]]
			sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

def predict(network, row):
	outputs = forward_propagate(network, row)
	if len(outputs) > 1:
		return outputs.index(max(outputs))
	else:
		return outputs[0]

x = [
		[0,0,1, 0],
		[0,1,1, 1],
		[1,0,1, 0],
		[1,1,1, 1],
	]

n_inputs = len(x[0]) - 1
n_outputs = 1
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, x, 0.5, 2000, n_outputs)

print('new example must be 0', predict(network, [0,0,1]))
print('new example must be 1', predict(network, [0,1,1]))
