import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from softmax import *


#learing rate = 0.1,batch_size = 500, epoch=30, lamda = 0.01
logging = runSoftmax(0.01,500,30,0.01)
plt.plot(logging[:,3],marker='+',label='learning rate = 0.1')


plt.legend(loc='lower right')
plt.title('Plot of Test_Accuracy vs. Iterations with batch_size=500')
plt.xlabel('Iterations')
plt.ylabel('Test_Accuracy')
plt.ylim(0,1)
plt.show()


