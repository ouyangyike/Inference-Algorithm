import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from linear import *

#learing rate = 0.001,batch_size = 100, epoch=31, lamda = 1
logging = runLinear(0.001,100,31,1)
plt.plot(logging[:,0],marker='*',label='learning rate = 0.001')

#learing rate = 0.01,batch_size = 100, epoch=31, lamda = 1
logging = runLinear(0.01,100,31,1)
plt.plot(logging[:,0],marker='d',label='learning rate = 0.01')

#learing rate = 0.1,batch_size = 100, epoch=31, lamda = 1
logging = runLinear(0.1,100,31,1)
plt.plot(logging[:,0],marker='h',label='learning rate = 0.1')


plt.legend(loc='upper right')
plt.title('Plot of Train_MSE vs. Iterations with batch_size=100')
plt.xlabel('Iterations')
plt.ylabel('Train_MSE')
plt.show()
