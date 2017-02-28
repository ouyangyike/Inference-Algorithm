import numpy as np
import tensorflow as tf


def loadData():
    # Loading my data
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.
        Target = Target[randIndx].reshape(-1,1)
        trainData, trainTarget = Data[:15000], Target[:15000].astype(np.int32)
        validData, validTarget = Data[15000:16000], Target[15000:16000].astype(np.int32)
        testData, testTarget = Data[16000:], Target[16000:].astype(np.int32)

    # reshape the data
    trainData = trainData.reshape((15000, -1))
    validData = validData.reshape((1000, -1))
    testData = testData.reshape((2724, -1))

    return trainData, trainTarget, validData, validTarget, testData, testTarget

def distribution(y_target):
    #compute the distribution of y_target
    #return a distribution matrix, where the indices denotes the label with the value=1,other value=0
    batch_size = tf.size(y_target)
    #y_target = tf.expand_dims(y_target,1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1,[indices, y_target])
    onehot_y_target = tf.sparse_to_dense(concated, [batch_size, 10], 1.0, 0.0)

    return onehot_y_target


def softmax(X, W, b, y_target, lamda):
    # compute the linear regression function and mean squared error
    # X:input data, [any value*784]
    # W:weights, [784*10]
    # y_predicted:prediction,[any value*10]

    #compute the prediction (posterior probability)
    y_predicted = tf.matmul(X, W) + b

    #compute the distribution of y_target
    onehot_y_target = distribution(y_target)

    #compute the crossEntropy
    crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= onehot_y_target,logits= y_predicted))+ 0.5 * lamda * tf.reduce_sum(tf.square(W))
    #0.5 * lamda * tf.reduce_sum(tf.reduce_mean(tf.square(W), 1))

    return y_predicted, crossEntropy


def buildGraph(learingRate, lamda):
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[784, 10], stddev=0.5, dtype=tf.float64), name='weights')
    b = tf.Variable(tf.truncated_normal(shape=[1, 10], stddev=0.5, dtype=tf.float64), name='biases')
    #b = tf.Variable(tf.zeros((1,10),dtype=tf.float64), expected_shape=[1, 10] , name='biases', dtype=tf.float64)
    X = tf.placeholder(tf.float64, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.int32, [None,1], name='target_y')

    # Graph and Error definition
    y_predicted, crossEntropy = softmax(X, W, b, y_target, lamda)

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate=learingRate)
    train = optimizer.minimize(loss=crossEntropy)
    return W, b, X, y_target, y_predicted, crossEntropy, train


def evaluate(y_target, y_predicted):
     # compute classification accuracy
     y_target = tf.reshape(y_target,[-1])
     correct = tf.nn.in_top_k(y_predicted, y_target, 1)
     accuracy = tf.reduce_sum(tf.cast(correct, tf.float64))/tf.cast(tf.size(correct),tf.float64)

     return accuracy


def runSoftmax(learningRate, batch_size, epoch, lamda):
    # Build computation graph
    W, b, X, y_target, y_predicted, crossEntropy, train = buildGraph(learningRate, lamda)

    # Loading my data
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData()

    # Initialize session
    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init)

    initialW = sess.run(W)
    initialb = sess.run(b)

    # set for logging
    #j = 0

    # Training model
    logging = np.zeros((epoch , 4))

    for step in range(0, epoch):
        # random shuffle trainData and trainTarget
        train_pre_random = np.concatenate((trainData, trainTarget), axis=1)
        np.random.shuffle(train_pre_random)
        trainData_random = train_pre_random[:, :784]
        trainTarget_random = train_pre_random[:, 784:785].astype(np.int32)

        # trainprocess on training set
        for i in range(0, len(trainTarget) // batch_size):
            _, train_err, currentW, currentb, y_train_predicted = sess.run([train, crossEntropy, W, b, y_predicted],feed_dict={X: trainData_random[(i * batch_size):((i + 1) * batch_size), :],y_target: trainTarget_random[(i * batch_size):((i + 1) * batch_size),:]})
        train_accuracy = sess.run(evaluate(trainTarget_random[(i * batch_size):((i + 1) * batch_size), :], y_train_predicted))

        # # valid set prediction
        # y_valid_predicted, valid_err = sess.run(softmax(validData, currentW, currentb, validTarget, lamda))
        # valid_accuracy = evaluate(validTarget, y_valid_predicted)

        # test set prediction
        y_test_predicted, test_err = sess.run(softmax(testData, currentW, currentb, testTarget, lamda))
        test_accuracy = sess.run(evaluate(testTarget, y_test_predicted))

        logging[step] = [train_err, test_err, train_accuracy, test_accuracy]


    return logging

