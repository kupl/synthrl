import numpy as np

from synthrl.value.value import Value

class BitVector(Value):
  SIZE = None

  def __init__(self, value=0):
    if isinstance(value, BitVector):
      value = value.get_value()
    if isinstance(value, str):
      value = int(value)
    elif not isinstance(value, int):
      raise ValueError('{} is not an integer.'.format(value))
    self.value = value

  def get_value(self):
    return self.value

  @classmethod
  def sample(cls):
    return cls()
