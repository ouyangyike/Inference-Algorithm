import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from kmeans import *

#learing rate = 0.1, K = 3, epoch=150
logging, percentage= runKmeans(0.1,5,300)
print percentage


#Loss on train set
plt.plot(logging[:,0],marker='+',label='K=3')
plt.legend(loc='upper right')
plt.title('Plot of Train_Loss vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Train_Loss')
plt.show()

