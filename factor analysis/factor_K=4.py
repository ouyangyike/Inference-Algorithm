import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from factor import *

#learing rate = 0.1, K = 4, D = 64, epoch=150
logging,currentw = runMoG(0.1,4,64,150)
plotWeight(currentw)


#Loss on train set
plt.figure(5)
plt.plot(logging[:,0],marker='+',label='Train_Loss')
plt.plot(logging[:,1],marker='+',label='Valid_Loss')
plt.plot(logging[:,2],marker='+',label='Test_Loss')
plt.legend(loc='upper right')
plt.title('Plot of Marginal log likelihood vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Marginal log likelihood')
plt.show()


