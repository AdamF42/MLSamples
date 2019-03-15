import numpy as np
import matplotlib.pyplot as plt

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

def graph(formula, x_range):
  x = np.array(x_range)
  y = formula(x)
  plt.plot(x, y)
  plt.show()

graph(lambda x: ReLU(x), range(-6, 6))
