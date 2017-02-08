import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from linear import *

#learing rate = 0.1,batch_size = 10, epoch=3, lamda = 1
logging = runLinear(0.1,10,3,1)
plt.plot(logging[:,0],marker='D',label='batch_size = 10')

#learing rate = 0.1,batch_size = 50, epoch=16, lamda = 1
logging = runLinear(0.1,50,16,1)
plt.plot(logging[:,0],marker='*',label='batch_size = 50')

#learing rate = 0.1,batch_size = 100, epoch=31, lamda = 1
logging = runLinear(0.1,100,31,1)
plt.plot(logging[:,0],marker='d',label='batch_size = 100')

#learing rate = 0.1,batch_size = 700, epoch=201, lamda = 1
logging = runLinear(0.1,700,201,1)
plt.plot(logging[:,0],marker='h',label='batch_size = 700')


plt.legend(loc='upper right')
plt.title('Plot of Train_MSE vs. Iterations with different batch_size')
plt.xlabel('Iterations')
plt.ylabel('Train_MSE')
plt.show()
