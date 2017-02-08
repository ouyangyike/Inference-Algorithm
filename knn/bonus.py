import tensorflow as tf
import numpy as np
from knn_module import *

def D(x, z):
    sess = tf.InteractiveSession()
    y = tf.squared_difference(x[:, tf.newaxis], z)
    result = tf.reduce_sum(y, 2)
    return result


def soft_knn(x, z):
    lamda = 10
    dist = tf.exp(-lamda * D(x, z))
    dist_sum = tf.reduce_sum(dist, 1)
    res = tf.div(dist, dist_sum[:, np.newaxis])
    return dist, res


def gaussian_process_regression(x, z):
    lamda = 100
    kernel = tf.matrix_inverse(tf.exp(-lamda*D(z, z)))
    dist = tf.exp(-lamda*D(x, z))
    return tf.matmul(dist, kernel)


def main():
    sess = tf.InteractiveSession()
    k = 1
    trainData, trainTarget, validData, validTarget, testData, testTarget, Data, Target = datagen()

    res_matrix = soft_knn(testData, trainData)[1]
    yexpected = tf.matmul(sess.run(res_matrix), trainTarget)
    print("yexpected:\n", sess.run(yexpected))

    res_matrix_gaussian = gaussian_process_regression(testData, trainData)
    yexpected_gaussian = tf.matmul(sess.run(res_matrix_gaussian), trainTarget)
    print("yexpected_gaussian:", sess.run(yexpected_gaussian))

    plt.plot(Data, Target, "rs", label='Data')
    plt.plot(testData, sess.run(yexpected), "bs", label='Soft_Knn')
    plt.plot(testData, sess.run(yexpected_gaussian), "gs", label='Gaussian_Knn')

    #plt.plot(X, sess.run(yexpected3),  marker='d', label='k = 3')
    #plt.plot(testData, yexpected,)

    plt.title('Expected Output vs. X')
    plt.xlabel('testData')
    plt.ylabel('Y/Y_expected Output')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
   main()
