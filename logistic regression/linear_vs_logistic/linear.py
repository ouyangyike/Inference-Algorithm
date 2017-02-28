import numpy as np
import tensorflow as tf


def loadData():
    # Loading my data
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx] / 255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500].astype(np.float64)
        validData, validTarget = Data[3500:3600], Target[3500:3600].astype(np.float64)
        testData, testTarget = Data[3600:], Target[3600:].astype(np.float64)

    # reshape the data
    trainData = trainData.reshape((3500, -1))
    validData = validData.reshape((100, -1))
    testData = testData.reshape((145, -1))

    return trainData, trainTarget, validData, validTarget, testData, testTarget


def MSE(X,W,b,y_target):
    # compute the linear regression function and mean squared error
    y_predicted = tf.matmul(X,W)+b
    meanSquareError = 0.5*tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target), reduction_indices=1, name='squared_error'), name='mean_squared_error')
    return y_predicted, meanSquareError


def buildGraph(learingRate):
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[784,1], stddev=0.5,dtype=tf.float64), name='weights')
    b = tf.Variable(0.0, name='biases',dtype=tf.float64)
    X = tf.placeholder(tf.float64, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float64, [None,1], name='target_y')

    # Graph and Error definition
    y_predicted, meanSquareError = MSE(X,W,b,y_target)

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = learingRate)
    train = optimizer.minimize(loss=meanSquareError)
    return W, b, X, y_target, y_predicted, meanSquareError, train

def evaluate(y_target, y_predicted):
    #compute classification accuracy
    
    j=0
    y_predicted = (y_predicted >= 0.5).astype(np.int)
    
    for i in xrange(len(y_target)):
        if y_target[i] == y_predicted[i]:
            j += 1
    accuracy = (j*1.0)/len(y_target)
    return accuracy


def runLinear(learningRate,batch_size, epoch):

    # Build computation graph
    W, b, X, y_target, y_predicted, meanSquareError, train = buildGraph(learningRate)

    # Loading my data
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData()
    
    # Initialize session
    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init)

    initialW = sess.run(W)  
    initialb = sess.run(b)
    
    #set for logging
    j=0
        
    # Training model
    logging = np.zeros((epoch*len(trainTarget)/batch_size,6))
    
    for step in xrange(0,epoch):
        #random shuffle trainData and trainTarget
        train_pre_random = np.concatenate((trainData, trainTarget),axis=1)
        np.random.shuffle(train_pre_random)
        trainData_random = train_pre_random[:,:784]
        trainTarget_random = train_pre_random[:,784:785]

        #trainprocess on training set
        for i in xrange(0,len(trainTarget)/batch_size):

            _, train_err, currentW, currentb, y_train_predicted = sess.run([train, meanSquareError, W, b, y_predicted], feed_dict={X: trainData_random[(i*batch_size):((i+1)*batch_size),:], y_target: trainTarget_random[(i*batch_size):((i+1)*batch_size),:]})
            train_accuracy = evaluate(trainTarget_random[(i * batch_size):((i + 1) * batch_size), :], y_train_predicted)

            #valid set prediction
            y_valid_predicted, valid_err = sess.run(MSE(validData,currentW,currentb,validTarget))
            valid_accuracy = evaluate(validTarget, y_valid_predicted)

            #test set prediction
            y_test_predicted, test_err = sess.run(MSE(testData,currentW,currentb,testTarget))
            test_accuracy = evaluate(testTarget, y_test_predicted)
        
            logging[j] = [train_err, valid_err, test_err, train_accuracy, valid_accuracy, test_accuracy]
            j+=1


    return logging

