import logging
import numpy as np

from synthrl.value.value import Value

logger = logging.getLogger(__name__)

class Integer(Value):
  MIN = -255
  MAX = 255
  def __init__(self, value=0):
    if isinstance(value, str):
      value = int(value)
    elif isinstance(value, float):
      value = int(value)
    elif not isinstance(value, int):
      raise ValueError('{} is not integer.'.format(value))
    elif value < Integer.MIN or value > Integer.MAX:
      logger.warning('The given value {} is not in between {} and {}. The value will be clipped.'.format(value, Integer.MIN, Integer.MAX))
      value = max(Integer.MIN, min(value, Integer.MAX))
    self.value = value

  def get_value(self):
    return self.value

  @classmethod
  def sample(cls):
    return cls(np.random.randint(cls.MIN, cls.MAX + 1))
