import dynet as dy
from utils import *
import network


if __name__ == "__main__":

	model = dy.Model()
	rnn = network.Network(W2I, I2W, model)
	#model.populate("model_wiki.m")
	#rnn.train(TRAIN, DEV)
	#rnn.evaluate(DEV)
