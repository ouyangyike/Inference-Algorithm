import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from MoG_valid import *

#learing rate = 0.1, K = 3, D=2, epoch=150
logging, indices = runMoG(0.1,5,2,300)
trainData,validData = loadData()


#Loss on train set
plt.figure(1)
plt.plot(logging[:,1],marker='+',label='K=5')
plt.legend(loc='upper right')
plt.title('Plot of Valid_Loss vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Valid_Loss')

plt.figure(2)
plt.scatter(validData[:,0],validData[:,1],c= indices,label='K=5')
plt.legend(loc='upper right')
plt.title('Scatter Plot of ValidData')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
