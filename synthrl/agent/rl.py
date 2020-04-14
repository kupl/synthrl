import torch.nn as nn
import torch.nn.functional as F

from synthrl.agent.agent import Agent
from synthrl.utils.torchutils import Attention

class Network(nn.Module):
  def __init__(self, input_dim=0, num_tokens=0, hidden_size=256, hidden_layers=1):
    super(Network, self).__init__()

    self.input_dim = input_dim
    self.num_tokens = num_tokens
    self.hidden_size = hidden_size
    self.hidden_layers = hidden_layers

    self.lstm = nn.LSTM(input_dim, hidden_size, hidden_layers, bidirectional=False)
    self.attention = Attention(hidden_size)
    self.policy = nn.Linear(hidden_size, num_tokens)
    self.value = nn.Linear(hidden_size, 1)

  def forward(self, inputs=None, hidden=None, outputs=None):
    #  inputs: [batch_size, n_example, input_dim]
    #  hidden: (hn, cn)
    #      hn: [1, batch_size * n_example, hidden_size]
    #      cn: [1, batch_size * n_example, hidden_size]
    # outputs: [batch_size * n_example, seq_len, hidden_size]
    batch_size, n_example, _ = inputs.shape

    # inputs: [1, batch_size * n_example, input_dim]
    inputs = inputs.reshape(-1, self.input_dim).unsqueeze(0)

    # query: [1, batch_size * n_example, hidden_size]
    query, hidden = self.lstm(inputs, hidden)

    # attn: [1, batch_size * n_example, hidden_size]
    attn = self.attention(query, outputs)

    # attn: [batch_size, hidden_size, n_example]
    attn = attn.reshape(batch_size, n_example, -1).permute(0, 2, 1)

    # pooled: [batch_size, hidden_size]
    pooled = F.max_pool1d(n_example).squeeze(-1)

    # policy: [batch_size, num_tokens]
    #  value: [batch_size, 1]
    policy = self.policy(pooled)
    value = self.value(pooled)

    policy = F.softmax(policy, dim=-1)
    value = -F.softplus(value) + self.eta

    return policy, value, query, hidden

class RLAgent(Agent):
  def __init__(self):
    self.network = None

  def take(self, action_space=[]):
    pass

  def save(self):
    pass

  def load(self):
    pass
