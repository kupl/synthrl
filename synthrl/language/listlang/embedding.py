import torch.nn as nn

class Embedding(nn.Module):
  def __init__(self, dimension=20):
    super(Embedding, self).__init__()

    self.dimension = dimension

  def forward(self, inputs, output):
    pass
