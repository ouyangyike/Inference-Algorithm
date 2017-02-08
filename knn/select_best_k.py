import tensorflow as tf
import numpy as np
from knn_module import *

def select_best_k(Data, Target, trainData, trainTarget, testTarget):
    sess = tf.InteractiveSession()
    X = np.linspace(0.0, 11.0, num=1000)[:, np.newaxis]

    knearest1 = run_knn(1, trainData, X)
    yexpected1 = tf.matmul(knearest1, trainTarget)

    knearest3 = run_knn(3, trainData, X)
    yexpected3 = tf.matmul(knearest3, trainTarget)

    knearest5 = run_knn(5, trainData, X)
    yexpected5 = tf.matmul(knearest5, trainTarget)

    knearest7 = run_knn(7, trainData, X)
    yexpected7 = tf.matmul(knearest7, trainTarget)

    knearest9 = run_knn(9, trainData, X)
    yexpected9 = tf.matmul(knearest9, trainTarget)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(Data, Target, marker='.', label='Data')
    ax1.plot(X, sess.run(yexpected1),  marker='*', label='k = 1')
    ax1.set_title('Expected Output vs. X')
    ax1.set_ylabel('Y/Y_expected Output')
    ax1.set_xlabel('X: 1000 equally spaced points between 0.0 and 11.0')
    ax1.legend(loc='lower right')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(Data, Target, marker='.', label='Data')
    ax2.plot(X, sess.run(yexpected3),  marker='d', label='k = 3')
    ax2.set_title('Expected Output vs. X')
    ax2.set_ylabel('Y/Y_expected Output')
    ax2.set_xlabel('X: 1000 equally spaced points between 0.0 and 11.0')
    ax2.legend(loc='lower right')

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(Data, Target, marker='.', label='Data')
    ax3.plot(X, sess.run(yexpected5), marker='h', label='k = 5')
    ax3.set_title('Expected Output vs. X')
    ax3.set_ylabel('Y/Y_expected Output')
    ax3.set_xlabel('X: 1000 equally spaced points between 0.0 and 11.0')
    ax3.legend(loc='lower right')

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(Data, Target, marker='.', label='Data')
    ax4.plot(X, sess.run(yexpected7), marker='o', label='k = 7')
    ax4.set_title('Expected Output vs. X')
    ax4.set_ylabel('Y/Y_expected Output')
    ax4.set_xlabel('X: 1000 equally spaced points between 0.0 and 11.0')
    ax4.legend(loc='lower right')

    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    ax5.plot(Data, Target, marker='.', label='Data')
    ax5.plot(X, sess.run(yexpected9), marker='o', label='k = 9')
    ax5.set_ylabel('Y/Y_expected Output')
    ax5.set_title('Expected Output vs. X')
    ax5.set_xlabel('X: 1000 equally spaced points between 0.0 and 11.0')
    ax5.legend(loc='lower right')


    plt.show()


if __name__ =="__main__":
    trainData, trainTarget, validData, validTarget, testData, testTarget, Data, Target \
        = datagen()

    select_best_k(Data, Target, trainData, trainTarget, testTarget)