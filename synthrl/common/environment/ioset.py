import numpy as np

class IOSet:

  def __init__(self, ioset):
    self.ioset = list(ioset)
    self.n_example = len(self.ioset)

  def get_rand(self, n):
    indices = np.random.permutation(self.n_example)[:n]
    return [self.ioset[i] for i in indices]
    
  def __getitem__(self, idx):
    return self.ioset[idx]

  def add(self, input, output):
    self.ioset.append((input, output))
    self.n_example += 1
