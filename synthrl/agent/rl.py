import torch
import torch.nn as nn
import torch.nn.functional as F

from synthrl.agent.agent import Agent
from synthrl.utils.torchutils import Attention

class Network(nn.Module):
  def __init__(self, num_tokens=0, token_embedding=20, num_values=0, value_embedding=20, hidden_size=256, hidden_layers=1):
    super(Network, self).__init__()

    self.num_tokens = num_tokens
    self.token_embedding = token_embedding
    self.num_values = num_values
    self.value_embedding = value_embedding
    self.hidden_size = hidden_size
    self.hidden_layers = hidden_layers

    self.token_embedding = nn.Embedding(num_tokens, token_embedding)
    self.value_embedding = nn.Embedding(num_values, value_embedding)
    self.lstm = nn.LSTM(token_embedding + value_embedding * 2, hidden_size, hidden_layers, bidirectional=False)
    self.attention = Attention(hidden_size)
    self.policy = nn.Linear(hidden_size, num_tokens)
    self.value = nn.Linear(hidden_size, 1)

  def forward(self, token=None, inputs=None, outputs=None):
    # token  : (batch_size, 1)
    # inputs : (batch_size, num_examples, ?)
    # outputs: (batch_size, num_examples, ?)
    raise NotImplementedError

class RLAgent(Agent):
  def __init__(self):
    self.policy_net = None
    self.value_net = None

  def take(self, action_space=[]):
    pass

  def save(self):
    pass

  def load(self):
    pass
