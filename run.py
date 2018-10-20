from csv import reader
from neural_network import NeuralNetwork


def read_csv_file(file_path):
	dataset = []
	with open(file_path, 'r') as file:
		for row in reader(file):
			if row:
				for col in range(len(row)):
					if col == len(row) - 1:
						row[col] = int(row[col].strip())
					else:
						row[col] = float(row[col].strip())
				dataset.append(row)
	return dataset


def normalize(dataset):
	stats = []
	for column_index, column in enumerate(zip(*dataset)):
		if column_index == len(dataset[0]) - 1:
			break
		stat = {}
		stat['min'] = min(column)
		stat['max'] = max(column)
		stat['average'] = sum(column) / len(column)
		stats.append(stat)
	for row in dataset:
		for col, col_stat in enumerate(stats):
			row[col] = (row[col] - col_stat['average']) / (col_stat['max'] - col_stat['min'])
	return dataset


dataset = read_csv_file('dataset.csv')
dataset = normalize(dataset)
n_inputs = len(dataset[0]) - 1
n_outputs = 3
network = NeuralNetwork(dataset, n_inputs, n_outputs)
network.train(1000)
