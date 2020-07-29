import torch
import torch.nn as nn

from synthrl.language.abstract import Bag
from synthrl.language.abstract import Embedding as EmbeddingInterface
from synthrl.language.bitvector import BitVectorLang
from synthrl.value.bitvector import BitVector

# integer represent None
NONE = BitVector.size

class TokenBag(Bag):

  @classmethod
  def create(cls):

    # create token to index bag
    bag = {t: i + 1 for i, t in enumerate(sorted(BitVectorLang.tokens))}
    bag['HOLE'] = 0
    return cls(bag)

def preprocess_value(inputs, output):
  # inputs: tuple (or list) of input
  # output: output value
  
  # make unsigned
  inputs = [i.unsigned for i in inputs]
  output = output.unsigned

  # marginalize with None to 2
  inputs = inputs + [NONE] * (2 - len(inputs))

  # return variables
  values = inputs + [output]

  return values

class Embedding(EmbeddingInterface):

  def __init__(self, token_dim=15, value_dim=80):
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
    self.n_value = BitVector.size + 1
    self.value_dim = value_dim
    # value_emb: [batch_size] -> [batch_size, value_dim]
    self.value_emb = nn.Embedding(self.n_value, self.value_dim)

  # result embedding dimension
  @property
  def emb_dim(self):
    return self.token_dim + (2 + 1) * self.value_dim

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

        # values: [3] of unsigned bitvector
        values = preprocess_value(input, output)

        # values: [3] of long tensor
        values = torch.LongTensor(values)

        # emb: [3, value_dim] of tensor
        emb = self.value_emb(values)

        # emb: [3 * value_dim]
        emb = emb.reshape(-1)

        io.append(emb)
    
    # io: [batch_size * n_example, 3 * value_dim]
    io = torch.stack(io)

    # io: [batch_size, n_example, 3 * value_dim]
    io = io.reshape(batch_size, -1, 3 * self.value_dim)
    n_example = io.shape[1]

    # token: [batch_size, 1, token_dim]
    token = token.unsqueeze(1)

    # token: [batch_size, n_example, token_dim]
    token = token.repeat(1, n_example, 1)

    # embedding: [batch_size, n_example, token_dim + 3 * value_dim]
    embedding = torch.cat((token, io), dim=2)

    # embedding: [batch_size, n_example, emb_dim]
    return embedding
