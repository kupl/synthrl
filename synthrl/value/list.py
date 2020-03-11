import logging
import numpy as np

from synthrl.value import Integer
from synthrl.value.value import Value

class List(Value):
  MAX_LENGTH = 256
  TYPE = None
  def __init__(self, value=[]):
    if not isinstance(value, list):
      raise ValueError('{} is not list.'.format(value))
    for v in value:
      if not isinstance(v, self.TYPE):
        raise ValueError('{} is not an instance of {}'.format(v, self.TYPE))
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

class IntList(List):
  TYPE = Integer
