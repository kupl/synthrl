from abc import ABC
from abc import abstractmethod
from pathlib import Path
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import importlib
import numpy as np
# pylint: disable=no-member
import torch
import torch.nn as nn
import torch.nn.functional as F

from synthrl.common.function.function import Function
from synthrl.common.utils import classproperty

PAD_TOKEN = '<pad>'


class Network(nn.Module):

  def __init__(self, input_size, hidden_size, n_layers, n_tokens):
    super(Network, self).__init__()

    # Layers.
    self.rnn = nn.LSTM(input_size, hidden_size, n_layers)
    self.policy = nn.Linear(hidden_size, n_tokens)
    self.value = nn.Linear(hidden_size, 1)

  def forward(self, inputs, lengths):
    # inputs: [batch_size, lengths, n_example, input_size]
    # lengths: [batch_size] of length tensor
    batch_size, max_len, n_example, input_size = inputs.shape

    # inputs: [batch_size, n_example, max_len, input_size]
    inputs = inputs.permute(0, 2, 1, 3)

    # inputs: [batch_size * n_example, max_len, input_size]
    inputs = inputs.reshape(-1, max_len, input_size)

    # lengths: [batch_size, 1]
    lengths = lengths.unsqueeze(-1)

    # lengths: [batch_size, n_example]
    lengths = lengths.repeat(1, n_example)

    # lengths: [batch_size * n_example]
    lengths = lengths.flatten()

    # inputs: packed input
    inputs = pack_padded_sequence(inputs, lengths)

    # outputs: packed pad outputs
    outputs = self.rnn(inputs)

    # outputs: [batch_size * n_example, max_len, hidden_size] of tensor
    # lengths: [batch_size * n_example] of length tensor
    outputs, lengths = pad_packed_sequence(outputs, batch_first=True)

    # outputs: [batch_size * n_example, hidden_size]
    outputs = torch.stack([outputs[batch, length - 1] for batch, length in enumerate(lengths)])

    # outputs: [batch_size, n_example, hidden_size]
    outputs = outputs.reshape(batch_size, n_example, -1)

    # outputs: [batch_size, hidden_size, n_example]
    outputs = outputs.permute(0, 2, 1)

    # pooled: [batch_size, hidden_size, 1]
    pooled = F.max_pool1d(outputs, n_example)

    # pooled: [batch_size, hidden_size]
    pooled = pooled.squeeze(-1)

    # policy: [batch_size, n_tokens]
    # value: [batch_size, 1]
    policy = self.policy(pooled)
    value = self.value(pooled)

    # policy: [batch_size, n_tokens]
    policy = torch.softmax(policy, dim=-1)

    # value: [batch_size, 1]
    value = torch.sigmoid(value)

    return policy, value


class RNNFunction(Function):

  def __init__(self, language, token_emb_dim, value_emb_dim, hidden_size, n_layers, device='cpu'):
    super(RNNFunction, self).__init__(language)
    
    # Padding token for RNN.
    self.tokens.append(PAD_TOKEN)
    self.indices[PAD_TOKEN] = len(self.tokens) - 1
    
    # Token embeding.
    self.n_tokens = len(self.tokens)
    self.token_emb_dim = token_emb_dim
    self.token_emb = nn.Embedding(self.n_tokens, self.token_emb_dim, padding_idx=self.indices[PAD_TOKEN]).to(device)

    # Value embeding.
    self.n_value_types = self.language.VALUE.N_VALUE
    self.value_emb_dim = value_emb_dim
    self.value_emb = nn.Embedding(self.n_value_types, self.value_emb_dim).to(device)

    # Main network.
    self.n_input = self.language.N_INPUT
    self.hidden_size = hidden_size
    self.n_layers = n_layers
    self.network = Network(token_emb_dim + (self.n_input + 1) * value_emb_dim, self.hidden_size, self.n_layers, self.n_tokens)

    # Other settings.
    self.device = device

  def evaluate(self, states, **info):
    # states: [batch_size] of tuple of (program, ioset)
    batch_size = len(states)

    #   pgms: [batch_size] of program
    # iosets: [batch_size] of ioset
    pgms, iosets = tuple(zip(*states))

    # pgms: [batch_size] of token sequence
    pgms = [pgm.sequence for pgm in pgms]

    # lengths: [batch_size] of length tensor
    lengths = torch.LongTensor([len(pgm) for pgm in pgms]).to(self.device)

    # Find descending order.
    # lengths: [batch_size] of lengths
    # sorted_idx: [batch_size] of indices
    lengths, sorted_idx = lengths.sort(descending=True)
    
    # Pad sequence.
    # max_len: maximum_length of sequence
    max_len = lengths.max()

    # pgms: [batch_size, max_len] of tokens
    pgms = [pgm + [PAD_TOKEN] * (max_len - len(pgm)) for pgm in pgms]

    # pgms: [batch_size, max_len] of token index tensor
    pgms = torch.LongTensor([[self.indices[token] for token in pgm] for pgm in pgm]).to(self.device)

    # Re-order in descending order.
    # pgms: [batch_size, max_len]
    pgms = pgms[sorted_idx]

    # Embed program.
    # pgms: [batch_size, max_len, token_emb_dim] of tensor
    pgms = self.token_emb(pgms)

    # iosets: [batch_size, n_example, n_input + 1] of value
    iosets = [[[*inputs, output] for inputs, output in ioset] for ioset in iosets]

    # iosets: [batch_size, n_example, n_input + 1] of value index tensor
    iosets = torch.LongTensor([[[value.index for value in example] for example in ioset] for ioset in iosets]).to(self.device)

    # Re-order in descending order.
    # iosets: [batch_size, n_example, n_input + 1]
    iosets = iosets[sorted_idx]

    # Embed ioset.
    # iosets: [batch_size, n_example, n_input + 1, value_emb_dim]
    iosets = self.value_emb(iosets)
    n_example = iosets.shape[1]

    # iosets: [batch_size, n_example, (n_input + 1) * value_emb_dim]
    iosets = iosets.reshape(batch_size, n_example, -1)

    # Expand dimension of token embedding
    # pgms: [batch_size, max_len, 1, token_emb_dim]
    pgms = pgms.unsqueeze(2)

    # pgms: [batch_size, max_len, n_example, token_emb_dim]
    pgms = pgms.repeat(1, 1, n_example, 1)

    # Expand dimension of ioset embedding
    # iosets: [batch_size, 1, n_example, (n_input + 1) * value_emb_dim]
    iosets = iosets.unsqeeze(1)

    # iosets: [batch_size, max_len, n_example, (n_input + 1) * value_emb_dim]
    iosets = iosets.repeat(1, max_len, 1, 1)
  
    # Concatenate token and ioset.
    # inputs: [batch_size, max_len, n_example, token_emb_dim + (n_input + 1) * value_emb_dim]
    inputs = torch.cat([pgms, iosets], dim=-1)

    # policy: [batch_size, n_tokens]
    # value: [batch_size, 1]
    policy, value = self.network(inputs, lengths)

    return policy, value

  def to(self, device):
    self.device = device
    self.token_emb.to(self.device)
    self.value_emb.to(self.device)
    self.network.to(self.device)
    return self

  def parameters(self):
    return self.token_emb.parameters() + self.value_emb.parameters() + self.network.parameters()

  def state_dict(self):
    return {
      'token_emb_state_dict': self.token_emb.state_dict(),
      'value_emb_state_dict': self.value_emb.state_dict(),
      'network_state_dict': self.network.state_dict()
    }

  def load_state_dict(self, state_dict):
    self.token_emb.load_state_dict(state_dict['token_emb_state_dict'])
    self.value_emb.load_state_dict(state_dict['value_emb_state_dict'])
    self.network.load_state_dict(state_dict['network_state_dict'])
    return self
  
  def save(self, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {
      'language': self.language.__name__,
      'token_emb_dim': self.token_emb_dim,
      'value_emb_dim': self.value_emb_dim,
      'hidden_size': self.hidden_size,
      'n_layers': self.n_layers,
      'state_dict': self.state_dict()
    }
    torch.save(info, path)


  @classmethod
  def load(cls, path, device='cpu'):
    path = Path(path)
    info = torch.load(path)
    language_module = importlib.import_module('synthrl.common.language')
    language = getattr(language_module, info['language'])
    function = cls(language, info['token_emb_dim'], info['value_emb_dim'], info['hidden_size'], info['n_layers'], device)
    function.load_state_dict(info['state_dict'])
    return function
    