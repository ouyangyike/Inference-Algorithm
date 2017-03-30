import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from MoG import *

#learing rate = 0.1, K = 3, D=2, epoch=300
logging, indices, currentmu, currentsigma, currentpi= runMoG(0.1,3,2,300)
trainData = loadData()
print indices
print currentmu
print currentsigma
print currentpi


#Loss on train set
plt.figure(1)
plt.plot(logging[:,0],marker='+',label='K=3')
plt.legend(loc='upper right')
plt.title('Plot of Train_Loss vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Train_Loss')

plt.figure(2)
plt.scatter(trainData[:,0],trainData[:,1],c= indices,label='scatter')
plt.legend(loc='upper right')
plt.title('Scatter Plot of TrainData')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

