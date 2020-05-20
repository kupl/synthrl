import torch
import torch.nn as nn

from synthrl.language.abstract import Bag
from synthrl.language.abstract import Embedding as EmbeddingInterface
from synthrl.language import ListLang
from synthrl.language.listlang.lang import S
from synthrl.value import Integer
from synthrl.value import IntList

# integer represent None
NONE = Integer.MAX + 2020

class TokenBag(Bag):

  @classmethod
  def create(cls):

    # create token to index bag
    bag = {t: i + 1 for i, t in enumerate(sorted(ListLang.tokens))}
    bag['HOLE'] = 0
    return cls(bag)

class ValueBag(Bag):

  @classmethod
  def create(cls):

    # create token to index bag
    bag = {i: idx + 1 for idx, i in range(Integer.MIN, Integer.MAX + 1)}
    bag[NONE] = 0
    return cls(bag)

def margin_value(inputs, output):
  # inputs: tuple (or list) of input
  # output: output value
  
  # marginalize with None to S
  inputs = list(inputs) + [NONE] * (S - len(inputs))

  # return variables
  types = []
  lists = []

  # process inputs
  for i in inputs:
    # if Integer
    if type(i) == Integer:
      # [1, 0]: Integer
      types.append([1, 0])
      # convert integer to list with margin of None
      lists.append([i.get_value()] + [NONE] * (IntList.MAX_LENGTH - 1))
    
    # if IntList
    elif type(i) == IntList:
      # [0, 1]: IntList
      types.append([0, 1])
      # marginalize with None to IntList.MAX_LENGTH
      lists.append(i.get_value() + [NONE] * IntList.MAX_LENGTH - len(i))

    # if None
    else:
      # [0, 0]: None
      types.append([0, 0])
      # append with None list
      lists.append([NONE] * IntList.MAX_LENGTH)

  # process output
  # if Integer
  if type(output) == Integer:
    # [1, 0]: Integer
    types.append([1, 0])
    # convert integer to list with margin of None
    lists.append([output.get_value()] + [NONE] * (IntList.MAX_LENGTH - 1))
  
  # if IntList
  elif type(output) == IntList:
    # [0, 1]: IntList
    types.append([0, 1])
    # marginalize with None to IntList.MAX_LENGTH
    lists.append(output.get_value() + [NONE] * IntList.MAX_LENGTH - len(output))

  # if None
  else:
    # [0, 0]: None
    types.append([0, 0])
    # append with None list
    lists.append([NONE] * IntList.MAX_LENGTH)

  return types, lists

class Embedding(EmbeddingInterface):

  def __init__(self, token_dim=15, value_dim=20):
    # token_dim: Embedding dimension of tokens.
    # value_dim: Embedding dimension of inputs and outputs.

    # initialize super class
    super(Embedding, self).__init__()

    # create token embedding layer
    self.token_bag = TokenBag.create()
    self.n_token = len(self.token_bag)
    self.token_dim = token_dim
    # token_emb: [batch_size] -> [batch_size, token_dim]
    self.token_emb = nn.Embedding(self.n_token, self.token_dim)

    # create i/o embedding layer
    self.value_bag = ValueBag.create()
    self.n_value = len(self.value_bag)
    self.value_dim = value_dim
    # value_emb: [batch_size] -> [batch_size, value_dim]
    self.value_emb = nn.Embedding(self.n_value, self.value_dim)

  # result embedding dimension
  @property
  def emb_dim(self):
    return self.token_dim + (S + 1) * (2 + IntList.MAX_LENGTH * self.value_dim)

  def forward(self, token, inputs, outputs):
    # token  : [batch_size]            of tokens
    # inputs : [batch_size, n_example] of tuples (or lists)
    # outputs: [batch_size, n_example] of values

    # embed token
    # token: [batch_size] of long tensor
    token = self.token_bag(token)

    # token: [batch_size, token_dim] of tensor
    token = self.token_emb(token)
    batch_size = token.shape[0]

    # embed inputs and outputs
    io = []
    for ioset in zip(inputs, outputs):
      # ioset: ([n_example] inputs, [n_example] outputs)

      for input, output in zip(*ioset):
        # input : 1 input (tuple, or list)
        # output: 1 output

        # types: [S + 1, 2]                  of int
        # lists: [S + 1, IntList.MAX_LENGTH] of int list
        types, lists = margin_value(input, output)

        # types: [S + 1, 2] of tensor
        types = torch.FloatTensor(types)
        
        # lists: [S + 1, IntList.MAX_LENGTH] of long tensor
        lists = torch.LongTensor(lists)

        # lists: [S + 1, IntList.MAX_LENGTH, value_dim] of tensor
        lists = self.value_emb(lists)

        # lists: [S + 1, IntList.MAX_LENGTH * value_dim]
        lists = lists.reshape(S + 1, -1)

        # emb: [S + 1, 2 + IntList.MAX_LENGTH * value_dim]
        embedding = torch.cat((types, lists), dim=1)

        # emb: [io_dim] = [(S + 1) * (2 + IntList.MAX_LENGTH * value_dim)]
        emb = emb.reshape(-1)
        io_dim = emb.shape[0]

        io.append(emb)
    
    # io: [batch_size * n_example, io_dim]
    io = torch.stack(io)

    # io: [batch_size, n_example, io_dim]
    io = io.reshape(batch_size, -1, io_dim)
    n_example = io.shape[1]

    # token: [batch_size, 1, token_dim]
    token = token.unsqueeze(1)

    # token: [batch_size, n_example, token_dim]
    token = token.repeat(1, n_example, 1)

    # embedding: [batch_size, n_example, token_dim + io_dim]
    embedding = torch.cat((token, io), dim=2)

    # embedding: [batch_size, n_example, emb_dim]
    return embedding
