import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from softmax import *



# #learing rate = 1,batch_size = 500, epoch=30, lamda = 0.01
# logging = runSoftmax(1,500,30,0.01)
# plt.plot(logging[:,2],marker='h',label='learning rate = 1')
#
# #learing rate = 0.1,batch_size = 500, epoch=30, lamda = 0.01
# logging = runSoftmax(0.1,500,30,0.01)
# plt.plot(logging[:,2],marker='+',label='learning rate = 0.1')

#learing rate = 0.01,batch_size = 500, epoch=30, lamda = 0.01
logging = runSoftmax(0.01,500,30,0.01)
plt.plot(logging[:,2],marker='+',label='learning rate = 0.01')


plt.legend(loc='lower right')
plt.title('Plot of Train_Accuracy vs. Iterations with batch_size=500')
plt.xlabel('Iterations')
plt.ylabel('Train_Accuracy')
plt.ylim(0,1)
plt.show()


