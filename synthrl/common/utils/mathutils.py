import numpy as np

def normalize(xs):
  xs = np.array(xs)
  return xs / xs.sum()
