import numpy as np

class IOSet:

  def __init__(self, ioset):
    self.ioset = list(ioset)
    self.n_example = len(self.ioset)

  def __len__(self):
    return self.n_example

  def get_rand(self, n):
    indices = (np.random.permutation(self.n_example).tolist() * (n // self.n_example + 1))[:n]
    return [self.ioset[i] for i in indices]
    
  def __getitem__(self, idx):
    return self.ioset[idx]

  def add(self, input, output):
    self.ioset.append((input, output))
    self.n_example += 1
