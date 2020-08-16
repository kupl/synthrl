from abc import ABC
from abc import abstractmethod

class Value(ABC):

  def __init__(self, value):
    self.value = value

  @classmethod
  @abstractmethod
  def sample(cls):
    pass

  @abstractmethod
  def __eq__(self, other):
    pass

  def __ne__(self, other):
    return not self == other

  def __str__(self):
    return repr(self)

  def __repr__(self):
    return repr(self.value)
