import numpy as np
import tensorflow as tf
from matplotlib.mlab import PCA
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

def generateData():
    tf.set_random_seed(521)
    s1 = tf.random_normal(shape=[200,1])
    s2 = tf.random_normal(shape=[200,1])
    s3 = tf.random_normal(shape=[200,1])
    x1 = s1
    x2 = s1 + 0.001*s2
    x3 = 10*s3
    data = tf.concat(values = [x1,x2,x3],concat_dim=1)

    return data

trainData = sess.run(generateData())
dataPCA = PCA(trainData)
var1 = sess.run(tf.reduce_sum(tf.square(dataPCA.Y[:,0]-trainData[:,0]))/200)
var2 = sess.run(tf.reduce_sum(tf.square(dataPCA.Y[:,0]-trainData[:,1]))/200)
var3 = sess.run(tf.reduce_sum(tf.square(dataPCA.Y[:,0]-trainData[:,2]))/200)
print ('variance of x1',var1)
print ('variance of x2',var2)
print ('variance of x3',var3)