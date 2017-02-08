import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from e122 import *

def main():
    sess = tf.InteractiveSession()
    trainData, trainTarget ,validData, validTarget, testData, testTarget,Data, Target = datagen()
    #test for knn
    run_knn(1, trainData, testData)

    #test for predict
    #testData, testTarget is what you want to replace
    #k = 50
    yexpected = predict(trainData, testData, trainTarget, testTarget, 1)[1]
    print("youtput:\n", sess.run(yexpected))

    #test for get_opt_k
    k_train = get_opt_k(trainData, trainData, trainTarget, trainTarget)
    #k_valid = get_opt_k(trainData, validData, trainTarget, validTarget)
    #k_test = get_opt_k(trainData, testData, trainTarget, testTarget)

    #test for draw, k_valid = 1
    k_draw(Data, Target, trainData, trainTarget, testTarget)


# Find the k value [1,3,5,50] with the minimum MSE
def get_opt_k(trainData, testData, trainTarget, testTarget):
    sess = tf.InteractiveSession()
    mse_result = []
    y_result = []
    k_min = 1
    min_mse_loss = sess.run( predict(trainData, testData, trainTarget, testTarget, 1)[0])
    y_expected = sess.run(predict(trainData, testData, trainTarget, testTarget, 1)[1])
    y_result.append(y_expected)
    mse_result.append(min_mse_loss)

    for k in [3, 5, 50]:
        mse_loss = sess.run(predict(trainData, testData, trainTarget, testTarget, k)[0])
        y_expected = sess.run(predict(trainData, testData, trainTarget, testTarget, k)[1])
        y_result.append(y_expected)
        mse_result.append(mse_loss)

        if mse_loss < min_mse_loss:
            min_mse_loss = mse_loss
            k_min = k

    print("min_k value is: ", k_min, "with mse_min of :", min_mse_loss)
    print("min_matrix for k 1, 3, 5, 50:\n", mse_result)
    print("y_result values for k 1, 3, 5, 50:\n", y_result)

    return k


def k_draw(Data, Target, trainData, trainTarget, testTarget):
    sess = tf.InteractiveSession()
    X = np.linspace(0.0, 11.0, num=1000)[:, np.newaxis]

    knearest1 = run_knn(1, trainData, X)
    yexpected1 = tf.matmul(knearest1, trainTarget)

    knearest3 = run_knn(3, trainData, X)
    yexpected3 = tf.matmul(knearest3, trainTarget)

    knearest5 = run_knn(5, trainData, X)
    yexpected5 = tf.matmul(knearest5, trainTarget)

    knearest50 = run_knn(50, trainData, X)
    yexpected50 = tf.matmul(knearest50, trainTarget)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(Data, Target, marker='.', label='Data')
    ax1.plot(X, sess.run(yexpected1),  marker='*', label='k = 1')
    ax1.set_ylabel('Y/Y_expected Output')
    ax1.set_xlabel('X')
    ax1.legend(loc='lower right')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(Data, Target, marker='.', label='Data')
    ax2.plot(X, sess.run(yexpected3),  marker='d', label='k = 3')
    ax2.set_ylabel('Y/Y_expected Output')
    ax2.set_xlabel('X')
    ax2.legend(loc='lower right')

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(Data, Target, marker='.', label='Data')
    ax3.plot(X, sess.run(yexpected5), marker='h', label='k = 5')
    ax3.set_ylabel('Y/Y_expected Output')
    ax3.set_xlabel('X')
    ax3.legend(loc='lower right')

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(Data, Target, marker='.', label='Data')
    ax4.plot(X, sess.run(yexpected50), marker='o', label='k = 50')
    ax4.set_ylabel('Y/Y_expected Output')
    ax4.set_xlabel('X')
    ax4.legend(loc='lower right')

    plt.show()


# given k, trainData, and trainTarget
# return yexpected with X input
def get_yexpected(k, trainData, trainTarget, X):
    knearest1 = run_knn(1, trainData, X)
    yexpected1 = tf.matmul(knearest1, trainTarget)
    return yexpected1


# return mse_loss, yexpected
def predict(trainData, testData, trainTarget, testTarget, k):
    sess = tf.InteractiveSession()
    knearest = run_knn(k, trainData, testData)

    yexpected = tf.matmul(knearest, trainTarget)
    sd = tf.squared_difference(yexpected, testTarget)

    ssd = tf.reduce_sum(sd)
    mse_loss = tf.divide(ssd, 2 * testTarget.shape[0])

    return (mse_loss, yexpected)


def get_euclidean_distance(x, z):
    y = tf.squared_difference(x[:, tf.newaxis], z)
    result = tf.reduce_sum(y, 2)
    return result


# return Response Matrix
def run_knn(k, trainData, testData):
    x1_ = tf.placeholder('float32')
    z1_ = tf.placeholder('float32')
    D_ = get_euclidean_distance(x1_, z1_)
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()

    dis = sess.run(D_, feed_dict={x1_: testData, z1_: trainData})
    dis *= -1

    res = tf.nn.top_k(dis, k)
    num_data_ptr = testData.shape[0]
    num_col = trainData.shape[0]

    n_reshape = sess.run(res.indices).reshape(num_data_ptr*k)

    responsibility_matrix = np.zeros([num_data_ptr, num_col])
    index_array = np.linspace(0, np.subtract(num_data_ptr, 1), num_data_ptr, dtype=int)
    # index_array: [0,1,2] index_array.repeat(2):[0,0,1,1,2,2]
    index_array = index_array.repeat(k)

    # reshape to 1D
    index_array = index_array.reshape(num_data_ptr*k)
    #print("index_array\n", index_array)

    responsibility_matrix[index_array, n_reshape] = float(1)/ k
    return responsibility_matrix


def datagen():
    np.random.seed(521)
    Data = np.linspace(1.0, 10.0 , num =100) [:, np. newaxis]
    Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
             + 0.5 * np.random.randn(100 , 1)
    # randIdx from 0,1,...99
    randIdx = np.arange(100)
    np.random.shuffle(randIdx)
    trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
    validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
    testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

    return (trainData, trainTarget,validData,
            validTarget, testData, testTarget
            ,Data, Target)


if __name__ =="__main__":
    main()