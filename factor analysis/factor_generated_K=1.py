import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from factor_generated import *

#learing rate = 0.1, K = 1, D = 3, epoch=150
logging,currentw = runMoG(0.01,1,3,5000)
print currentw




