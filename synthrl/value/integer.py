import logging
import numpy as np

from synthrl.value.value import Value

logger = logging.getLogger(__name__)

class Integer(Value):
  MIN = -255
  MAX = 255

  def __init__(self, value=0):
    if isinstance(value, Integer):
      value = value.get_value()
    elif isinstance(value, str):
      value = int(value)
    elif isinstance(value, float):
      value = int(value)
    elif not isinstance(value, int):
      raise ValueError('{} is not an integer.'.format(value))
    elif value < Integer.MIN or value > Integer.MAX:
      logger.warning('The given value {} is not in between {} and {}. The value will be clipped.'.format(value, Integer.MIN, Integer.MAX))
      value = max(Integer.MIN, min(value, Integer.MAX))
    self.value = value

  def get_value(self):
    return self.value

  @classmethod
  def sample(cls):
    return cls(np.random.randint(cls.MIN, cls.MAX + 1))

  def __index__(self):
    return self.value

  def __neg__(self):
    return Integer(-self.get_value())

  def __add__(self, other):
    if not isinstance(other, Integer):
      raise ValueError('Operator + is not supported between Integer and {}'.format(other.__class__.__name__))
    return Integer(self.get_value() + other.get_value())

  def __sub__(self, other):
    if not isinstance(other, Integer):
      raise ValueError('Operator - is not supported between Integer and {}'.format(other.__class__.__name__))
    return Integer(self.get_value() - other.get_value())

  def __mul__(self, other):
    if not isinstance(other, Integer):
      raise ValueError('Operator * is not supported between Integer and {}'.format(other.__class__.__name__))
    return Integer(self.get_value() * other.get_value())

  def __pow__(self, other):
    if not isinstance(other, Integer):
      raise ValueError('Operator ** is not supported between Integer and {}'.format(other.__class__.__name__))
    return Integer(self.get_value() ** other.get_value())

  def __truediv__(self, other):
    if not isinstance(other, Integer):
      raise ValueError('Operator / is not supported between Integer and {}'.format(other.__class__.__name__))
    return Integer(self.get_value() // other.get_value())

  def __floordiv__(self, other):
    if not isinstance(other, Integer):
      raise ValueError('Operator // is not supported between Integer and {}'.format(other.__class__.__name__))
    return Integer(self.get_value() // other.get_value())

  def __eq__(self, other):
    if not isinstance(other, Integer):
      raise ValueError('Operator == is not supported between Integer and {}'.format(other.__class__.__name__))
    return self.get_value() == other.get_value()

  def __ne__(self, other):
    if not isinstance(other, Integer):
      raise ValueError('Operator != is not supported between Integer and {}'.format(other.__class__.__name__))
    return self.get_value() != other.get_value()

  def __lt__(self, other):
    if not isinstance(other, Integer):
      raise ValueError('Operator < is not supported between Integer and {}'.format(other.__class__.__name__))
    return self.get_value() < other.get_value()

  def __le__(self, other):
    if not isinstance(other, Integer):
      raise ValueError('Operator <= is not supported between Integer and {}'.format(other.__class__.__name__))
    return self.get_value() <= other.get_value()

  def __gt__(self, other):
    if not isinstance(other, Integer):
      raise ValueError('Operator > is not supported between Integer and {}'.format(other.__class__.__name__))
    return self.get_value() > other.get_value()

  def __ge__(self, other):
    if not isinstance(other, Integer):
      raise ValueError('Operator >= is not supported between Integer and {}'.format(other.__class__.__name__))
    return self.get_value() >= other.get_value()


# constants
ZERO = Integer(0)
ONE = Integer(1)
TWO = Integer(2)
THREE = Integer(3)
FOUR = Integer(4)
