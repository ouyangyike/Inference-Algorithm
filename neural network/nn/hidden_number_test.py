import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nn import *





#learing rate = 0.01,batch_size = 500, epoch=5, lamda = 3e-4, hidden_number= 500
logging = runNN(0.01,500,20,3e-4,500)
plt.plot(logging[:,5],marker='+',label='hidden number = 500')




plt.legend(loc='upper right')
plt.title('Plot of  Test_Error vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Test_Error')
plt.show()




