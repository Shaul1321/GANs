import numpy as np


with open("gen.txt", "r") as f:

	lines = f.readlines()

total = []

for line in lines:

	sent, vec = line.strip().split("\t")
	vec_str = vec.split(" ")
	vec = np.array([float(x) for x in vec_str])
	total.append(vec)

total = np.array(total)
means = np.average(total, axis = 0)
stds = np.std(total, axis = 0)
with open("stats.txt", "w") as f:

	f.write(",".join(['%.6f' % number for number in means]) + "\n")
	f.write(",".join(['%.6f' % number for number in stds]) + "\n")
