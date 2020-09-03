from abc import ABC
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

from synthrl.common.utils import classproperty

class Value(ABC):

  @abstractmethod
  @property
  def index(self):
    pass

class Program(ABC):

  @abstractmethod
  @classproperty
  @classmethod
  def tokens(cls):
    pass


class Embedding(nn.Module):

  def __init__(self, ):
    super(Embedding, self).__init__()

class AttentionBased(nn.Module):

  def __init__(self, ):
    super(AttentionBased, self).__init__()
