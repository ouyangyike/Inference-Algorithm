import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from linear import *


#learing rate = 0.1,batch_size = 50, epoch=16, lamda = 0
logging = runLinear(0.1,50,16,0)
plt.plot(logging[:,3],marker='*',label='lamda = 0.')

#learing rate = 0.1,batch_size = 50, epoch=16, lamda = 0.0001
logging = runLinear(0.1,50,16,0.0001)
plt.plot(logging[:,3],marker='d',label='lamda = 0.0001')

#learing rate = 0.1,batch_size = 50, epoch=16, lamda = 0.001
logging = runLinear(0.1,50,16,0.001)
plt.plot(logging[:,3],marker='h',label='lamda = 0.001')

#learing rate = 0.1,batch_size = 50, epoch=16, lamda = 0.01
logging = runLinear(0.1,50,16,0.01)
plt.plot(logging[:,3],marker='s',label='lamda = 0.01')

#learing rate = 0.1,batch_size = 50, epoch=16, lamda = 0.1
logging = runLinear(0.1,50,16,0.1)
plt.plot(logging[:,3],marker='+',label='lamda = 0.1')

#learing rate = 0.1,batch_size = 50, epoch=16, lamda = 1
logging = runLinear(0.1,50,16,1)
plt.plot(logging[:,3],marker='D',label='lamda = 1')



plt.legend(loc='lower right')
plt.title('Plot of Valid_Accuracy vs. Iterations with batch_size=50')
plt.xlabel('Iterations')
plt.ylabel('Valid_Accuracy')
plt.show()
