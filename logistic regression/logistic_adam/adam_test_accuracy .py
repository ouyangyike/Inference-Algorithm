import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from logistic_adam import *


#learing rate = 0.1,batch_size = 500, epoch=15, lamda = 0.01
logging = runLogistic(0.1,500,15,0.01)
#print(logging)
plt.plot(logging[:,5],marker='*',label='learning rate = 0.1')




plt.legend(loc='lower right')
plt.title('Plot of Test_Accuracy vs. Iterations with batch_size=500')
plt.xlabel('Iterations')
plt.ylabel('Test_Accuracy')
plt.ylim(0,1)
plt.show()
