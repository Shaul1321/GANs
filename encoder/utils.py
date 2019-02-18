import random
from collections import Counter
random.seed(1)

def read_sentences(path):

	with open(path, "r") as f:
	
		lines = []
		
		for i, line in enumerate(f):
			if i > 500000: break
			lines.append(line.strip())

	return lines

def read_voc(fname = "voc_wiki.txt"):

	words = []
			
	words.append("<S>")
	words.append("<E>")
	words.append("<unk>")
	
	with open(fname, "r") as f:
		lines = f.readlines()
		
	for line in lines:
	
		words.append(line.strip())


	return list(set(words))
	
def replace_unknown(lines, voc):

	for i, line in enumerate(lines):
	
		splitted = line.split(" ")
		for j, w in enumerate(splitted):
		
			if w not in voc:
			
				splitted[j] = "<unk>"
				
		lines[i] = " ".join(splitted)
		
	return lines

def load_file(fname):

	with open(fname, "r") as f:
	
		lines = f.readlines()
	lines = [line.strip().split(" ") for line in lines]
	return lines
			

voc = read_voc()
TRAIN = load_file("train_wiki.txt")
DEV = load_file("dev_wiki.txt")
TEST = load_file("test_wiki.txt")
VOC_SIZE = len(voc)
#del lines
#del voc
W2I = {w:i for (i,w) in enumerate(sorted(voc))}
I2W = {i:w for (i,w) in enumerate(sorted(voc))}
