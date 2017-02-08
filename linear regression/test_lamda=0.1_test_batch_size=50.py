import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from linear import *

#learing rate = 0.1,batch_size = 50, epoch=16, lamda = 0.1
logging = runLinear(0.1,50,16,0.1)
#print(logging)
plt.plot(logging[:,4],marker='+',label='lamda = 0.1')


plt.legend(loc='lower right')
plt.title('Plot of Test_Accuracy vs. Iterations with batch_size=50')
plt.xlabel('Iterations')
plt.ylabel('Test_Accuracy')
plt.show()
