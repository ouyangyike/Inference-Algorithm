import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nn_dropout import *


#learing rate = 0.01,batch_size = 500, epoch=5, lamda = 3e-4, hidden_number=1000
logging= runNN(0.01,500,3,3e-4,1000)

plt.plot(logging[:,3],marker='d',label='Train_Error')
plt.plot(logging[:,4],marker='h',label='Validation_Error')
plt.plot(logging[:,5],marker='*',label='Test_Error')


plt.legend(loc='upper right')
plt.title('Plot of Error vs. Iterations with batch_size=500')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()




