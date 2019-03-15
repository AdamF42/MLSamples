import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def graph(formula, x_range):
  x = np.array(x_range)
  y = formula(x)
  plt.plot(x, y)
  plt.show()

graph(lambda x: sigmoid(x), range(-6, 6))
