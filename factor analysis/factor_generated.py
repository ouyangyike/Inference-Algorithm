import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt


def generateData():
    tf.set_random_seed(1002704406)
    s1 = tf.random_normal(shape=[200,1])
    s2 = tf.random_normal(shape=[200,1])
    s3 = tf.random_normal(shape=[200,1])
    x1 = s1
    x2 = s1 + 0.001*s2
    x3 = 10*s3
    data = tf.cast(tf.concat(values = [x1,x2,x3],concat_dim=1),tf.float64)

    return data


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

    # Initialize session
    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init)

    initialw = sess.run(w)
    initialmu = sess.run(mu)
    initialsigma = sess.run(sigma)

    # Loading my data
    trainData = sess.run(generateData())

    # set for logging
    j = 0

    # Training model
    logging = np.zeros((epoch, 1))

    for step in range(0, epoch):
        # trainprocess on training set
        _,currentw, currentmu, currentsigma, trainLoss, currentDist = sess.run([train, w, mu, sigma, totalLoss, dist],feed_dict={X: trainData})

        logging[j] = [trainLoss]
        j += 1
    print ('Final train marginal log likelihood:',trainLoss)

    return logging, currentw

