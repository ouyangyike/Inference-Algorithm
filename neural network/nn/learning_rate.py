import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nn import *



#learing rate = 0.001,batch_size = 500, epoch=13, lamda = 3e-4, hidden_number= 1000
logging = runNN(0.001,500,20,3e-4,1000)
plt.plot(logging[:,0],marker='h',label='learning rate = 0.001')

#learing rate = 0.01,batch_size = 500, epoch=5, lamda = 3e-4, hidden_number= 1000
logging = runNN(0.01,500,20,3e-4,1000)
plt.plot(logging[:,0],marker='+',label='learning rate = 0.01')


#learing rate = 0.1,batch_size = 500, epoch=13, lamda = 3e-4, hidden_number= 1000
logging = runNN(0.1,500,20,3e-4,1000)
plt.plot(logging[:,0],marker='d',label='learning rate = 0.1')


plt.legend(loc='upper right')
plt.title('Plot of  Train_Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Train_Loss')
plt.show()




