import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt


def loadData():
    # Loading my data
    with np.load("tinymnist.npz") as data:
        trainData, trainTarget = data["x"], data["y"]
        validData, validTarget = data["x_valid"], data["y_valid"]
        testData, testTarget = data["x_test"], data["y_test"]

    return trainData, trainTarget, validData, validTarget, testData, testTarget


def Euclidean_dis(X, Y):
    # return the matrix containing the pairwise Euclidean distances
    X_b = tf.expand_dims(X, 1)
    result = X_b - Y
    result_square = tf.square(result)
    Euclidean_dis = tf.reduce_sum(result_square, 2)

    return Euclidean_dis

def logPdf(x,w,mu,sigma,D):
    # compute the log probability density function for cluster k
    #x: B*D
    #mu:k*D
    #sigma:1*k
    #logpdf:B*k
    distance = x-mu
    A = tf.diag(tf.exp(sigma)) + tf.matmul(w,tf.transpose(w))
    log_det = 2.0*tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(A))))
    logpdf = -0.5*log_det-D/2*tf.log(tf.cast(2.0*math.pi,tf.float64))-0.5*tf.diag_part(tf.matmul(tf.matmul(distance,tf.matrix_inverse(A)),tf.transpose(distance)))

    return logpdf

def LOSS(x,w,mu,sigma,D):
    # compute the loss function
    dist = logPdf(x,w,mu,sigma,D)
    loss = -tf.reduce_sum(dist)
    return loss, dist

def buildGraph(learingRate,K,D):
    # Variable creation
    tf.set_random_seed(521)
    w = tf.Variable(tf.random_normal(shape=[D,K],stddev=0.5,dtype=tf.float64),name = 'w')
    mu = tf.Variable(tf.random_normal(shape=[1,D],stddev=0.5,dtype=tf.float64),name = 'mu')
    sigma = tf.Variable(tf.random_normal(shape=[D,],stddev=0.5,dtype=tf.float64),name = 'sigma')
    X = tf.placeholder(dtype=tf.float64, shape=[None, D], name='trainData')

    # Graph and Error definition
    totalLoss, dist = LOSS(X,w,mu,sigma,D)

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = learingRate,beta1=0.9, beta2=0.99,epsilon=1e-5)
    train = optimizer.minimize(loss=totalLoss)
    return X, w, mu, sigma, totalLoss, dist, train


def runMoG(learningRate, K, D, epoch):
    # Build computation graph
    X, w, mu, sigma, totalLoss, dist, train = buildGraph(learningRate,K,D)

    # Loading my data
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData()

    # Initialize session
    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init)

    initialw = sess.run(w)
    initialmu = sess.run(mu)
    initialsigma = sess.run(sigma)

    # set for logging
    j = 0

    # Training model
    logging = np.zeros((epoch, 3))

    for step in range(0, epoch):
        # trainprocess on training set
        _,currentw, currentmu, currentsigma, trainLoss, currentDist = sess.run([train, w, mu, sigma, totalLoss, dist],feed_dict={X: trainData})

        validLoss, validDist = sess.run(LOSS(validData, currentw, currentmu, currentsigma, D))

        testLoss, testDist = sess.run(LOSS(testData, currentw, currentmu, currentsigma, D))

        logging[j] = [trainLoss,validLoss,testLoss]
        j += 1
    print ('Final train marginal log likelihood:',trainLoss)
    print ('Final valid marginal log likelihood:', validLoss)
    print ('Final test marginal log likelihood:', testLoss)

    return logging, currentw

def plotWeight(units):
    units = np.reshape(units,(8,8,4))
    for j in range(4):
        plt.figure(j+1)
        plt.imshow(units[:,:,j], interpolation="nearest", cmap="gray")
        plt.show()