import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from kmeans_valid import *

#learing rate = 0.1, K = 1, epoch=300
logging = runKmeans(0.1,1,300)
plt.plot(logging[:,1],marker='d',label='K=1')


#learing rate = 0.1, K = 2, epoch=300
logging = runKmeans(0.1,2,300)
plt.plot(logging[:,1],marker='*',label='K=2')


#learing rate = 0.1, K = 3, epoch=300
logging = runKmeans(0.1,3,300)
plt.plot(logging[:,1],marker='+',label='K=3')


#learing rate = 0.1, K = 4, epoch=300
logging = runKmeans(0.1,4,300)
plt.plot(logging[:,1],marker='h',label='K=4')


#learing rate = 0.1, K = 5, epoch=300
logging = runKmeans(0.1,5,300)
plt.plot(logging[:,1],marker='D',label='K=5')



plt.legend(loc='upper right')
plt.title('Plot of Valid_Loss vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Valid_Loss')
plt.show()