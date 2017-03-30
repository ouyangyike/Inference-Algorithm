import numpy as np
import tensorflow as tf

def loadData():
    # Loading my data
    trainData = np.load('data100D.npy')

    # random shuffle trainData and trainTarget
    np.random.seed(521)
    np.random.shuffle(trainData)
    validData_random = trainData[:3333, :]
    trainData_random = trainData[3334:, :]
    return trainData_random, validData_random

def LOSS(X,Mu):
    # compute the loss function
    #X:trainData [10000 2]
    #Mu: center value matrix [3 2]
    dist = tf.reduce_sum(tf.square(X[:, tf.newaxis] - Mu), 2)
    loss = tf.reduce_sum(tf.reduce_min(dist,1))
    return loss, dist

def evaluate(dist,K):
    percentage = []
    _, indices = tf.nn.top_k(-dist)
    indices = tf.reshape(indices,[-1])
    for i in xrange(K):
        data = tf.cast(i*tf.ones((tf.size(indices),)),tf.int32)
        correct = tf.equal(indices,data)
        percentage.append(tf.reduce_sum(tf.cast(correct, tf.float64)) / tf.cast(tf.size(indices),tf.float64))

    return percentage




def buildGraph(learingRate,K):
    # Variable creation
    Mu = tf.Variable(tf.random_normal(shape=[K,100],stddev=1.0,dtype=tf.float64),name = 'Mu')
    X = tf.placeholder(dtype=tf.float64, shape=[None, 100], name='trainData')

    # Graph and Error definition
    totalLoss, dist = LOSS(X,Mu)

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = learingRate,beta1=0.9, beta2=0.99,epsilon=1e-5)
    train = optimizer.minimize(loss=totalLoss)
    return X, Mu, totalLoss, train, dist


def runKmeans(learningRate, K, epoch):
    # Build computation graph
    X, Mu, totalLoss, train, dist = buildGraph(learningRate,K)

    # Loading my data
    trainData, validData = loadData()

    # Initialize session
    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init)

    initialMu = sess.run(Mu)

    # set for logging
    j = 0

    # Training model
    logging = np.zeros((epoch, 1))

    for step in xrange(0, epoch):
        # trainprocess on training set
        _,currentDist, currentMu, trainLoss = sess.run([train, dist, Mu, totalLoss],feed_dict={X: trainData})

        logging[j] = [trainLoss]
        j += 1

    validLoss, validDist = sess.run(LOSS(validData, currentMu))
    trainPercentage = sess.run(evaluate(currentDist,K))
    validPercentage = sess.run(evaluate(validDist,K))
    print ('final train loss:', trainLoss)
    print ('final valid loss:', validLoss)
    print ('final train percentage:', trainPercentage)
    print ('final valid percentage:', validPercentage)

    return logging
