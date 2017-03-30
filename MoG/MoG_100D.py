import numpy as np
import tensorflow as tf
import math
from utils import *

def loadData():
    # Loading my data
    trainData = np.load('data100D.npy')

    # random shuffle trainData and trainTarget
    np.random.seed(521)
    np.random.shuffle(trainData)
    validData_random = trainData[:3333, :]
    trainData_random = trainData[3334:, :]
    return trainData_random, validData_random


def Euclidean_dis(X, Y):
    # return the matrix containing the pairwise Euclidean distances
    X_b = tf.expand_dims(X, 1)
    result = X_b - Y
    result_square = tf.square(result)
    Euclidean_dis = tf.reduce_sum(result_square, 2)

    return Euclidean_dis

def logPdf(x,mu,sigma,D):
    # compute the log probability density function for cluster k
    #x: B*D
    #mu:k*D
    #sigma:1*k
    #logpdf:B*k
    distance = Euclidean_dis(x,mu)
    logpdf = -D/2*tf.log(2.0*math.pi*tf.square(sigma))-tf.div(tf.cast(distance,tf.float64),tf.cast(2.0*tf.square(sigma),tf.float64))
    return logpdf

def posterior(dist,pre_pi):
    #pre_pi: 1*k
    #dist: B*k
    #posterior: B*k
    posterior = logsoftmax(pre_pi)+dist-reduce_logsumexp(logsoftmax(pre_pi)+dist,1)[:,tf.newaxis]
    return posterior

def LOSS(x,mu,sigma,pre_pi,D):
    # compute the loss function

    dist = logPdf(x,mu,sigma,D)
    loss = -tf.reduce_sum(reduce_logsumexp(logsoftmax(pre_pi)+dist,1))
    return loss, dist

def evaluate(dist,K):
    percentage = []
    indices = tf.cast(tf.arg_max(dist,1),tf.int32)
    for i in xrange(K):
        data = tf.cast(i*tf.ones((tf.size(indices),)),tf.int32)
        correct = tf.equal(indices,data)
        percentage.append(tf.reduce_sum(tf.cast(correct, tf.float64)) / tf.cast(tf.size(indices),tf.float64))

    return percentage


def buildGraph(learingRate,K,D):
    # Variable creation
    tf.set_random_seed(521)
    mu = tf.Variable(tf.random_normal(shape=[K,D],stddev=0.5,dtype=tf.float64),name = 'mu')
    sigma = tf.Variable(tf.exp(tf.random_normal(shape=[1,K],stddev=0.5,dtype=tf.float64)),name = 'sigma')
    pre_pi = tf.Variable(tf.random_normal(shape=[1,K],stddev=0.5,dtype=tf.float64), name='pre_pi')
    X = tf.placeholder(dtype=tf.float64, shape=[None, D], name='trainData')

    # Graph and Error definition
    totalLoss, dist = LOSS(X,mu,sigma,pre_pi,D)

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = learingRate,beta1=0.9, beta2=0.99,epsilon=1e-5)
    train = optimizer.minimize(loss=totalLoss)
    return X, mu, sigma, pre_pi, totalLoss, dist, train


def runMoG(learningRate, K, D, epoch):
    # Build computation graph
    X, mu, sigma, pre_pi, totalLoss, dist, train = buildGraph(learningRate,K,D)

    # Loading my data
    trainData, validData = loadData()

    # Initialize session
    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init)

    initialmu = sess.run(mu)
    initialsigma = sess.run(sigma)
    initialpre_pi = sess.run(pre_pi)

    # set for logging
    j = 0

    # Training model
    logging = np.zeros((epoch, 1))

    for step in xrange(0, epoch):
        # trainprocess on training set
        _,currentmu, currentsigma, currentpre_pi, trainLoss, currentDist = sess.run([train, mu, sigma, pre_pi, totalLoss, dist],feed_dict={X: trainData})


        logging[j] = [trainLoss]
        j += 1
    trainPosterior = sess.run(posterior(currentDist,currentpre_pi))
    trainPercentage = sess.run(evaluate(trainPosterior,K))
    validLoss, validDist = sess.run(LOSS(validData, currentmu, currentsigma, currentpre_pi, D))
    validPosterior = sess.run(posterior(validDist, currentpre_pi))
    validPercentage = sess.run(evaluate(validPosterior,K))
    currentpi = sess.run(tf.exp(logsoftmax(currentpre_pi)))

    print ('final train loss:', trainLoss)
    print ('final valid loss:',validLoss)
    print ('final train percentage:', trainPercentage)
    print ('final valid percentage:', validPercentage)
    print ('final sigma:', currentsigma)
    print ('final pi:',currentpi)

    return logging
