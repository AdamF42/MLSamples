import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    z_exp = np.exp(z)
    sum_z_exp = np.sum(z_exp)
    return np.array([round(i/sum_z_exp, 3) for i in z_exp])

def graph(formula, x_range):
  x = np.array(x_range)
  y = formula(x)
  plt.plot(x, y)
  plt.show()

graph(lambda x: softmax(x), range(-6, 6))
