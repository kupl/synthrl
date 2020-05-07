import torch
import torch.nn as nn

# Abstract helper class for converting tokens into index
class Bag:

  __slots__ = ['bag', 'unbag']

  def __init__(self, bag):
    # bag: A dictionary that contains all tokens and their corresponding index.

    self.bag = bag
    self.unbag = {v: k for k, v in self.bag.items()}

  @classmethod
  def create(cls):
    # create a new bag object
    raise NotImplementedError

  def __call__(self, token):
    # gets tokens
    # token : [batch_size]

    # returns indices of tokens
    # return: [batch_size]

    # convert tokens into indices
    return torch.LongTensor([self.bag[t] for t in token])

  def inverse(self, index):
    # gets indices
    # index : [batch_size]

    # returns corresponding tokens of indices
    # return: [batch_size]

    # convert indices into tokens
    return [self.unbag[i] for i in index]

  def __len__(self):
    # returns number of tokens
    return len(self.bag)

  @classmethod
  def load_from(cls, dict):
    # gets a dictionary created by save_to
    # returns created bag

    # create and return a new bag from dict
    return cls(dict['bag'])

  def save_to(self):
    # returns a dictionary that contains information to a create identical bag

    # return necessary information as dictionary
    return {'bag': self.bag}

# Abstract class that each language should implement to create embedding
class Embedding(nn.Module):

  def __init__(self):
    # initialize the embedding layer
    # super().__init__() must be called
    super(Embedding, self).__init__()

  def forward(self, token, inputs, outputs):
    # gets token, inputs, outputs
    # token  : [batch_size]
    # inputs : [batch_size, n_example]
    # outputs: [batch_size, n_example]

    # returns concatenated embedding
    # return : [batch_size, n_example, embedding_dim]

    raise NotImplementedError
