import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from sklearn import datasets
import random

with open("encoding_vectors.txt", "r") as f:

    lines = f.readlines()


L = 10000
lines = lines[:L]


vecs = []
for line in lines:

    vec_str = line.strip().split(" ")
    vec = np.array([float(x) for x in vec_str])
    vecs.append(vec)

vecs = np.array(vecs)

def get_closest_vector(vec):

    similarity = vecs.dot(vec.T)
    i =  np.argmax(similarity)
    return words[i]
    
def get_data(n_points):
    """
     Returns a synthetic dataset.
    """
    
    chosens = []
    for i in range(n_points):
    
        chosen = random.choice(vecs)
        chosen /= np.linalg.norm(chosen)
        chosens.append(chosen)
        
    return np.array(chosens)
    
if __name__ == "__main__":

 X = get_data(15)
 print (X.shape)


