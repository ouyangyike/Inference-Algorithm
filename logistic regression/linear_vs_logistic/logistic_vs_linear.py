import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def MSE(y_predicted,y_target):
    meanSquareError = 0.5*tf.square(y_predicted - y_target)
    return meanSquareError

def CE(y_predicted, y_target):
    crossEntropy = -y_target*tf.log(y_predicted)-(1-y_target)*tf.log(1-y_predicted)
    #crossEntropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predicted,targets=y_target)
    return crossEntropy

sess = tf.InteractiveSession()

plt.plot(np.linspace(0,1,1000),sess.run(MSE(np.linspace(0,1,1000),0)),label='MSE')
plt.plot(np.linspace(0,1,1000),sess.run(CE(np.linspace(0,1,1000),np.zeros((1000,)))),label='CrossEntropy')
plt.title('Plot of Cross_Entropy vs. Mean_Square_Error')
plt.legend(loc='upper right')
plt.show()



