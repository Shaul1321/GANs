import dynet as dy
import random

EMBEDDING_SIZE = 256
from utils import VOC_SIZE
LSTM_HIDDENSIZE = 1024
DROPOUT = 0.0
PREDICTION_HIDDEN_LAYERS = [LSTM_HIDDENSIZE, 1024, 2048, VOC_SIZE]
NUM_LAYERS = 1
import numpy as np
import os
try:
 os.remove("gen.txt")
except OSError:
 pass
 
 
with open("stats.txt", "r") as f:

    lines = f.readlines()
    means, stds = lines[0].strip().split(","), lines[1].strip().split(",")
    MEANS = np.array([float(x) for x in means])
    STDS = np.array([float(x) for x in stds])

class Network(object):

     def __init__(self, tok2ind, ind2tok, model):
    
        self.tok2ind = tok2ind
        self.ind2tok = ind2tok
        self.model = model
        self.create_model()
        self.best_acc = -1

     def create_model(self):

     
        self.encoder = dy.LSTMBuilder(NUM_LAYERS, EMBEDDING_SIZE, LSTM_HIDDENSIZE, self.model)
        self.decoder = dy.LSTMBuilder(NUM_LAYERS, LSTM_HIDDENSIZE+EMBEDDING_SIZE, LSTM_HIDDENSIZE, self.model)
        self.prediction_weights = []
        
        params = []
            
        for (in_size, out_size) in zip(PREDICTION_HIDDEN_LAYERS, PREDICTION_HIDDEN_LAYERS[1:]):
    
                    W = self.model.add_parameters((out_size, in_size))
                    b = self.model.add_parameters((out_size, 1))
                    self.prediction_weights.append((W,b))
            
        self.E = self.model.add_lookup_parameters((len(self.tok2ind), EMBEDDING_SIZE))
        self.trainer = dy.AdamTrainer(self.model)

     def encode(self, x, bilstm = False):
        encoded_x = [self.E[self.tok2ind[w] if w in self.tok2ind else self.tok2ind["<unk>"]] for w in x]

        s = self.encoder.initial_state()
        states = s.transduce(encoded_x)
        last_state = states[-1]
        
        #v = dy.vecInput(LSTM_HIDDENSIZE)
        #cov = np.diag(STDS)
        #v.set(np.random.multivariate_normal(MEANS, cov))
        #return v
        return last_state
        
     def decode(self, encoded_state, x):
        
            s = self.decoder.initial_state()
            start_encoded = self.E[self.tok2ind["<S>"]]
            s = s.add_input(dy.concatenate([start_encoded, encoded_state]))
            loss = dy.scalarInput(0.)
            
            for w in x[1:]:
            
                true_word_encoded = self.E[self.tok2ind[w] if w in self.tok2ind else self.tok2ind["<unk>"]]
                probs, scores = self.predict_word(s.output())
                s = s.add_input(dy.concatenate([true_word_encoded, encoded_state]))
            
                loss += dy.pickneglogsoftmax(scores, self.tok2ind[w] if w in self.tok2ind else self.tok2ind["<unk>"])
            
            return loss
        
     def predict_word(self, hidden_state):

            h = hidden_state
            
            for k, (w, b) in enumerate(self.prediction_weights):
        
               w_param, b_param = dy.parameter(w), dy.parameter(b)
               h = w_param * h + b_param
                
               if k != len(self.prediction_weights) - 1:
                    
                  h = dy.rectify(h)
                    
            probs = dy.softmax(h)
            return probs, h
                
     def train(self, train_data, dev_data, num_epochs = 300, batch_size = 32):
        
            
        for I in range(num_epochs):
            
                avg_loss = 0.
                random.shuffle(train_data)
                good, bad = 0., 0.
                avg_edit_distance = 0.
                q = 0.
                   
                losses = []
                   
                for i, x in enumerate(train_data):

                    if i % batch_size == 0 and i > 0 and losses:
                        
                        loss_sum = dy.esum(losses)
                        loss_sum.forward()
                        loss_sum.backward()
                        self.trainer.update()
                        losses = []
                        dy.renew_cg()
                            
                    
                    encoded_state = self.encode(x)
                    
                    loss = self.decode(encoded_state, x)
                    losses.append(loss)
                    avg_loss += loss.value()
        
                    if i % 5000 == 0 and i > 0:
            
                        print ("evaluating accuracy on dev set.")
                        acc = self.evaluate(dev_data)  
                        losses = []
                        
                        if acc > self.best_acc:

                          self.best_acc = acc
                          self.model.save("model.m")
                        

                #self.embedding_collector.collect()
        
     def generate_from_encoding_vector(self, numpy_vec):
        
             with open("gen.txt", "a+") as f:
             
                 
                 dy.renew_cg()
                 gen = ["<S>"]
                 encoded = dy.vecInput(len(numpy_vec))
                 encoded.set(numpy_vec)
                 s = self.decoder.initial_state()
                 start_encoded = self.E[self.tok2ind["<S>"]]
                 s = s.add_input(dy.concatenate([start_encoded, encoded]))   
                 counter = 0
                 current = "<S>"
            
                 while counter < 50 and current != "<E>":
            
                    counter += 1
                    probs, scores = self.predict_word(s.output())
                    s = s.add_input(dy.concatenate([self.E[self.tok2ind[current]], encoded]))
                    current = self.ind2tok[np.argmax(probs.npvalue())]
                    gen.append(current)
                 
                 gen = " ".join(gen)    
                 f.write(gen + "\n")                
                 
     def evaluate(self, evalset):
            
            diff = 0.
            count = 0
            
            with open("gen.txt", "w") as f:
            
                for i, x in enumerate(evalset):
            
                    dy.renew_cg()
                    encoded = self.encode(x)
                    s = self.decoder.initial_state()
                    start_encoded = self.E[self.tok2ind["<S>"]]
                    s = s.add_input(dy.concatenate([start_encoded, encoded]))
            
                    gen = []
                
                    for w in x[1:]:
                
                        count += 1.
                        probs, scores = self.predict_word(s.output())
                        true_word_encoded = self.E[self.tok2ind[w] if w in self.tok2ind else self.tok2ind["<unk>"]]
                        s = s.add_input(dy.concatenate([true_word_encoded, encoded]))
                        predicted_word = self.ind2tok[np.argmax(probs.npvalue())]
                        gen.append(predicted_word)
                    
                        if predicted_word != w:      
                          
                            diff += 1
                    encoded_as_str = " ".join(['%.6f' % number for number in encoded.npvalue()])
                    #f.write(" ".join(gen) + "\t" +  encoded_as_str + "\n")
                    f.write(" ".join(gen) + "\n")
            
                acc = 1 - (diff / count)
                print (acc)
                return acc
