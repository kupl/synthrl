# pylint: disable=no-member
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
  def __init__(self, dimension=0):
    super(Attention, self).__init__()

    self.dimension = dimension

    self.linear_in = nn.Linear(dimension, dimension, bias=False)
    self.linear_out = nn.Linear(2 * dimension, dimension, bias=False)

  def forward(self, hidden=None, context=None):
    #  hidden: [1, batch_size, dimension]
    # context: [batch_size, seq_len, dimension]

    # hidden: [batch_size, dimension]
    hidden = self.linear_in(hidden)

    # hidden: [batch_size, 1, dimension]
    hidden = hidden.permute(1, 0, 2)

    # scores: [batch_size, 1, seq_len]
    scores = hidden.bmm(context.permute(0, 2, 1))
    scores = F.softmax(scores, dim=-1)

    # mix: [batch_size, 1, dimension]
    mix = scores.bmm(context)

    # combined: [batch_size, 1, 2 * dimension]
    combined = torch.cat((mix, hidden), dim=2)

    # outputs: [batch_size, 1, dimension]
    outputs = self.linear_out(combined)

    # outputs: [1, batch_size, dimension]
    outputs = outputs.permute(1, 0, 2)

    return outputs
