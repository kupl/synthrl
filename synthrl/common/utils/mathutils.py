import numpy as np

def normalize(xs, epsilon=1e-5):
  xs = np.array(xs)
  return xs / (xs.sum() + epsilon)
