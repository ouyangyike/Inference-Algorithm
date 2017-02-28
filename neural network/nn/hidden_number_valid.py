import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nn import *



#learing rate = 0.01,batch_size = 500, epoch=13, lamda = 0.01, hidden_number= 1000
logging = runNN(0.01,500,20,3e-4,1000)
plt.plot(logging[:,4],marker='h',label='hidden number = 1000')

#learing rate = 0.01,batch_size = 500, epoch=5, lamda = 3e-4, hidden_number= 500
logging = runNN(0.01,500,20,3e-4,500)
plt.plot(logging[:,4],marker='+',label='hidden number = 500')


#learing rate = 0.1,batch_size = 500, epoch=13, lamda = 3e-4, hidden_number= 100
logging = runNN(0.01,500,20,3e-4,100)
plt.plot(logging[:,4],marker='d',label='hidden number = 100')


plt.legend(loc='upper right')
plt.title('Plot of  Valid_Error vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Valid_Error')
plt.show()




