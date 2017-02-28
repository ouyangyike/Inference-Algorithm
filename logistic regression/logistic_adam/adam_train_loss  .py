import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from logistic_adam import *


#learing rate = 1,batch_size = 500, epoch=15, lamda = 0.01
logging = runLogistic(1,500,15,0.01)
#print(logging)
plt.plot(logging[:,0],marker='+',label='learning rate = 1')

#learing rate = 0.1,batch_size = 500, epoch=15, lamda = 0.01
logging = runLogistic(0.1,500,15,0.01)
#print(logging)
plt.plot(logging[:,0],marker='*',label='learning rate = 0.1')

#learing rate = 0.01,batch_size = 500, epoch=15, lamda = 0.01
logging = runLogistic(0.01,500,15,0.01)
#print(logging)
plt.plot(logging[:,0],marker='h',label='learning rate = 0.01')

#learing rate = 0.001,batch_size = 500, epoch=15, lamda = 0.01
logging = runLogistic(0.001,500,15,0.01)
#print(logging)
plt.plot(logging[:,0],marker='d',label='learning rate = 0.001')


plt.legend(loc='upper right')
plt.title('Plot of Train_CrossEntropy vs. Iterations with batch_size=500')
plt.xlabel('Iterations')
plt.ylabel('Train_CrossEntropy')
plt.show()


