import torch
import torch.nn as nn

from synthrl.language.abstract import Bag
from synthrl.language.abstract import Embedding as EmbeddingLayer
from synthrl.language import ListLang
from synthrl.value import Integer
from synthrl.value import IntList
from synthrl.value.nonetype import NONE

class TokenBag(Bag):

  @classmethod
  def create(cls):

    # create token to index bag
    bag = {t: i + 1 for i, t in enumerate(ListLang.tokens)}
    bag['HOLE'] = 0
    return cls(bag)

class ValueBag(Bag):

  @classmethod
  def create(cls):

    # create token to index bag
    bag = {Integer(i): idx + 1 for idx, i in range(Integer.MIN, Integer.MAX + 1)}
    bag[NONE] = 0
    return cls(bag)

class Embedding(EmbeddingLayer):

  def __init__(self, n_token, n_value, token_dim=15, value_dim=20):
    # token_dim: Embedding dimension of tokens.
    # value_dim: Embedding dimension of inputs and outputs.

    # initialize super class
    super(Embedding, self).__init__()

    # create token embedding layer
    self.n_token = n_token
    self.token_dim = token_dim
    # token_emb: [batch_size] -> [batch_size, token_dim]
    self.token_emb = nn.Embedding(self.n_token, self.token_dim)

    # create i/o embedding layer
    self.n_value = n_value
    self.value_dim = value_dim
    # value_emb: [batch_size] -> [batch_size, value_dim]
    self.value_emb = nn.Embedding(self.n_value, self.value_dim)

    # create type embedding
    # [1, 0]: Integer
    # [0, 1]: IntList
    # [0, 0]: None
    # type_emb: [batch_size] -> [batch_size, 2]
    self.type_emb = lambda batch: torch.tensor([[1 if type(x) == Integer else 0, 1 if type(x) == IntList else 0] for x in batch], dtype=torch.float64)

  def forward(self, token, inputs, outputs):
    # token  : [batch_size]
    # inputs : [batch_size, n_example]
    # outputs: [batch_size, n_example]
    
    raise NotImplementedError
