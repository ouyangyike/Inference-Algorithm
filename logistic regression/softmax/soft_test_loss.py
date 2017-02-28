import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from softmax import *



#learing rate = 0.01,batch_size = 500, epoch=30, lamda = 0.01
logging = runSoftmax(0.01,500,30,0.01)
plt.plot(logging[:,1],marker='d',label='learning rate = 0.01')


plt.legend(loc='upper right')
plt.title('Plot of Test_Loss vs. Iterations with batch_size=500')
plt.xlabel('Iterations')
plt.ylabel('Test_Loss')
plt.show()


