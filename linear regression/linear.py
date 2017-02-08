import numpy as np
import tensorflow as tf

def loadData():
    # Loading my data
    trainData, trainTarget = np.load("x.npy"), np.load("y.npy")
    validData, validTarget = np.load("x_valid.npy").astype(np.float32), np.load("y_valid.npy").astype(np.float32)
    testData, testTarget = np.load("x_test.npy").astype(np.float32), np.load("y_test.npy").astype(np.float32)
    
    return trainData, trainTarget, validData, validTarget, testData, testTarget

def MSE(X,W,b,y_target,lamda):
    # compute the linear regression function and mean squared error
    y_predicted = tf.matmul(X,W)+b
    meanSquareError = 0.5*tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target), reduction_indices=1, name='squared_error'), name='mean_squared_error')+0.5*lamda*tf.reduce_sum(tf.square(W))
    return y_predicted, meanSquareError


def buildGraph(learingRate,lamda):
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[64,1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 64], name='input_x')
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')

    # Graph and Error definition
    y_predicted, meanSquaredError = MSE(X,W,b,y_target,lamda)

    # Training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learingRate)
    train = optimizer.minimize(loss=meanSquaredError)
    return W, b, X, y_target, y_predicted, meanSquaredError, train

def evaluate(y_target, y_predicted):
    #compute classification accuracy
    
    j=0
    y_predicted = (y_predicted >= 0.5).astype(np.int)
    
    for i in xrange(len(y_target)):
        if y_target[i] == y_predicted[i]:
            j += 1
    accuracy = (j*1.0)/len(y_target)
    return accuracy


def runLinear(learningRate,batch_size, epoch, lamda):

    # Build computation graph
    W, b, X, y_target, y_predicted, meanSquaredError, train = buildGraph(learningRate,lamda)

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
    logging = np.zeros((epoch*len(trainTarget)/batch_size,5))
    
    for step in xrange(0,epoch):
        #random shuffle trainData and trainTarget
        train_pre_random = np.concatenate((trainData, trainTarget),axis=1)
        np.random.shuffle(train_pre_random)
        trainData_random = train_pre_random[:,:64]
        trainTarget_random = train_pre_random[:,64:65]

        #trainprocess on training set
        for i in xrange(0,len(trainTarget)/batch_size):

            _, train_err, currentW, currentb, y_train_predicted = sess.run([train, meanSquaredError, W, b, y_predicted], feed_dict={X: trainData_random[(i*batch_size):((i+1)*batch_size),:], y_target: trainTarget_random[(i*batch_size):((i+1)*batch_size),:]})

            #valid set prediction
            y_valid_predicted, valid_err = sess.run(MSE(validData,currentW,currentb,validTarget,lamda))
            valid_accuracy = evaluate(validTarget, y_valid_predicted)

            #test set prediction
            y_test_predicted, test_err = sess.run(MSE(testData,currentW,currentb,testTarget,lamda))
            test_accuracy = evaluate(testTarget, y_test_predicted)
        
            logging[j] = [train_err, valid_err, test_err, valid_accuracy, test_accuracy]
            j+=1

    return logging

