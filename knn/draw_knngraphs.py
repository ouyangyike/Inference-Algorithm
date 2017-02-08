import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from knn_module import *

def run_everything():
    sess = tf.InteractiveSession()
    trainData, trainTarget ,validData, validTarget, testData, testTarget,Data, Target = datagen()

    #test for knn
    run_knn(1, trainData, testData)

    #test for predict
    #testData, testTarget is what you want to replace
    yexpected = predict(trainData, testData, trainTarget, testTarget, k=1)[1]
    print("youtput:\n", sess.run(yexpected))

    #test for get_opt_k
    k_train = get_opt_k(trainData, trainData, trainTarget, trainTarget)
    k_valid = get_opt_k(trainData, validData, trainTarget, validTarget)
    k_test = get_opt_k(trainData, testData, trainTarget, testTarget)

    #test for draw, k_valid = 1
    k_draw(Data, Target, trainData, trainTarget, testTarget)

if __name__ =="__main__":
    run_everything()