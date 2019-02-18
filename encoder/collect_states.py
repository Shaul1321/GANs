import dynet as dy
from utils import *
import network


def collect_states(model, dataset):
	
	data = []
	
	for i, sentence in enumerate(dataset):
	
			dy.renew_cg()
			encoded = model.encode(sentence)
			encoded_as_str = " ".join(['%.6f' % number for number in encoded.npvalue()])
			data.append(encoded_as_str)
	return data

def print_states(data):
	
	with open("encoding_vectors.txt", "w") as f:
		for encoded_as_str in data:
	
			f.write(encoded_as_str + "\n")
	
if __name__ == "__main__":

	model = dy.Model()
	rnn = network.Network(W2I, I2W, model)
	model.populate("model.m")
	#rnn.train(TRAIN, DEV)
	states_data = collect_states(rnn, TRAIN)
	print_states(states_data)
	
