import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from logistic_adam import *
from linear import *


#learing rate = 0.1,batch_size = 500, epoch=15
logging_logistic = runLogistic(0.1,500,15)
logging_linear = runLinear(0.01,500,15)

#Loss on train set
plt.figure(1)
plt.plot(logging_logistic[:,0],marker='+',label='logistic')
plt.plot(logging_linear[:,0],marker='*',label='linear')
plt.legend(loc='upper right')
plt.title('Plot of Train_Loss vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Train_Loss')
plt.show()

#Loss on validation set
plt.figure(2)
plt.plot(logging_logistic[:,1],marker='+',label='logistic')
plt.plot(logging_linear[:,1],marker='*',label='linear')
plt.legend(loc='upper right')
plt.title('Plot of Valid_Loss vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Valid_Loss')
plt.show()

#Loss on test set
plt.figure(3)
plt.plot(logging_logistic[:,2],marker='+',label='logistic')
plt.plot(logging_linear[:,2],marker='*',label='linear')
plt.legend(loc='upper right')
plt.title('Plot of Test_Loss vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Test_Loss')
plt.show()





