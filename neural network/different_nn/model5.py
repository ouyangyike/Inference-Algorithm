import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nn5 import *

#seed = 19930107,704406,521521,100470,93521
seed = np.random.seed(93521)

#learning rate
learning_rate = np.exp(np.random.uniform(-7.5,-4.5,1))
print learning_rate
#weight decay
lamda = np.exp(np.random.uniform(-9,-6,1))
print lamda
#number of layers
layers = np.random.random_integers(1,5,1)
print layers
#number of units in each layer
units = np.random.random_integers(100,500,layers)
print units
#whether to use dropout
dropout = np.random.sample()>0.5
print dropout
#probability in dropout
probability = np.random.uniform()
print probability

#learing rate=0.00311976 ,batch_size = 500, epoch=6, lamda=0.00030087, hidden_number=[335]
#use dropout,p= 0.135554221007
valid_error, test_error = runNN(0.00311976,500,4,0.00030087,335)
print valid_error
print test_error