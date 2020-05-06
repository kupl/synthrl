from collections.abc import Iterable
import logging
import numpy as np

from synthrl.value.integer import Integer
from synthrl.value.value import Value

logger = logging.getLogger(__name__)

class List(Value):
  MAX_LENGTH = 20
  TYPE = Value
  def __init__(self, value=[]):
    if not isinstance(value, Iterable):
      raise ValueError('{} is not iterable.'.format(value))
    self.value = [self.TYPE(v) for v in value]
    if len(self.value) > self.MAX_LENGTH:
      logger.warning('The length of the given list is greater than {}. The elements over {} will be discarded.'.format(self.MAX_LENGTH, self.MAX_LENGTH))
      self.value = self.value[:self.MAX_LENGTH]

  def get_value(self):
    return [v.get_value() for v in self.value]

  @classmethod
  def sample(cls):
    length = np.random.randint(List.MAX_LENGTH)
    value = [cls.TYPE.sample() for _ in range(length)]
    return cls(value)

  def append(self, value):
    if not isinstance(value, self.TYPE):
      raise ValueError('{} is not an instance of {}'.format(value, self.TYPE))
    self.value.append(value)

  def __add__(self, other):
    return self.__class__(self.get_value() + other.get_value())

  def __iter__(self):
    for v in self.value:
      yield v

  def __getitem__(self, idx):
    if isinstance(idx, slice):
      return self.__class__(self.value[idx])
    else:
      return self.value[idx]

  def __len__(self):
    return len(self.value)

  def __reversed__(self):
    return self.__class__(reversed(self.get_value()))

  def __eq__(self, other):
    if not isinstance(other, self.__class__):
      if isinstance(other, Value):
        return False
      raise ValueError('Cannot compare {} and {}.'.format(self.__class__.__name__, other.__class__.__name__))
    if len(self) != len(other):
      return False
    for i, j in zip(self, other):
      if i != j:
        return False
    return True

  def __ne__(self, other):
    return not (self == other)

class IntList(List):
  TYPE = Integer
