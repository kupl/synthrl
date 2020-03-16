import logging
import numpy as np

from synthrl.value import Integer
from synthrl.value.value import Value

class List(Value):
  MAX_LENGTH = 256
  TYPE = Value
  def __init__(self, value=[]):
    if not isinstance(value, list):
      raise ValueError('{} is not list.'.format(value))
    value = [self.TYPE(v) for v in value]
    self.value = value

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
    return self.value[idx]

  def __len__(self):
    return len(self.value)

  def __reversed__(self):
    return self.__class__(list(reversed(self.get_value())))

class IntList(List):
  TYPE = Integer