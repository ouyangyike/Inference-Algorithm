import numpy as np
import matplotlib.pyplot as plt
from kmeans import *

def draw_scatter(k):
    logging, percentage, indices= runKmeans(0.1,k,300)
    trainData = loadData()

    plt.scatter(trainData[:,0],trainData[:,1],c=indices,label='K='+ str(k))
    plt.legend(loc='upper right')
    plt.title('Scatter Plot of TrainData')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

draw_scatter(1)
draw_scatter(2)
draw_scatter(3)
draw_scatter(4)
draw_scatter(5)