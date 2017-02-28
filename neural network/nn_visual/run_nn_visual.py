import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nn_visual import *


#learing rate = 0.01,batch_size = 500, epoch=5, lamda = 3e-4, hidden_number=1000
logging_hidden_w= runNN(0.01,500,20,3e-4,1000)
plotNNFilter(logging_hidden_w[19,:,:])






