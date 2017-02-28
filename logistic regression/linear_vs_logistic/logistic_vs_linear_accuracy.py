import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from logistic_adam import *
from linear import *


#learing rate = 0.1,batch_size = 500, epoch=20
logging_logistic = runLogistic(0.1,500,20)
logging_linear = runLinear(0.01,500,20)

#Loss on train set
plt.figure(1)
plt.plot(logging_logistic[:,3],marker='+',label='logistic')
plt.plot(logging_linear[:,3],marker='*',label='linear')
plt.legend(loc='lower right')
plt.title('Plot of Train_Accuracy vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Train_Accuracy')
plt.show()

#Loss on validation set
plt.figure(2)
plt.plot(logging_logistic[:,4],marker='+',label='logistic')
plt.plot(logging_linear[:,4],marker='*',label='linear')
plt.legend(loc='lower right')
plt.title('Plot of Valid_Acccuracy vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Valid_Accuracy')
plt.show()

#Loss on test set
plt.figure(3)
plt.plot(logging_logistic[:,5],marker='+',label='logistic')
plt.plot(logging_linear[:,5],marker='*',label='linear')
plt.legend(loc='lower right')
plt.title('Plot of Test_Accuracy vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Test_Accuracy')
plt.show()





