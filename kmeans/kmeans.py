import numpy as np
import tensorflow as tf

def loadData():
    # Loading my data
    trainData = np.load('data2D.npy')
    return trainData

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
        data = tf.cast(i*tf.ones((10000,)),tf.int32)
        correct = tf.equal(indices,data)
        percentage.append(tf.reduce_sum(tf.cast(correct, tf.float64)) / 10000)

    return percentage, indices




def buildGraph(learingRate,K):
    # Variable creation
    Mu = tf.Variable(tf.random_normal(shape=[K,2],stddev=1.0,dtype=tf.float64),name = 'Mu')
    X = tf.placeholder(dtype=tf.float64, sbuildGraphhape=[10000, 2], name='trainData')

    # Graph and Error definition
    totalLoss, dist = LOSS(X,Mu)

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = learingRate,beta1=0.9, beta2=0.99,epsilon=1e-5)
    train = optimizer.minimize(loss=totalLoss)
    return X, Mu, totalLoss, train, dist


def runKmeans(learningRate, K, epoch):
    # Build computation graph
    X, Mu, totalLoss, train, dist = (learningRate,K)

    # Loading my data
    trainData = loadData()

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


    percentage, indices = sess.run(evaluate(currentDist,K))
    print ('Final train loss:',trainLoss)

    return logging, percentage, indices
