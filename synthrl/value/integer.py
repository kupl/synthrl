import logging
import numpy as np

from synthrl.value.value import Value

logger = logging.getLogger(__name__)

class Integer(Value):
  MIN = -256
  MAX = 255

  def __init__(self, value=0):
    if isinstance(value, Integer):
      value = value.get_value()
    elif not (isinstance(value, int) or isinstance(value, str) or isinstance(value, float)):
      raise ValueError('{} is not an integer.'.format(value))
    elif value < Integer.MIN or value > Integer.MAX:
      logger.info('The given value {} is not in [{}, {}]. The value will be clipped.'.format(value, Integer.MIN, Integer.MAX))
      value = max(Integer.MIN, min(value, Integer.MAX))
    self.value = int(value)

  def get_value(self):
    return self.value

  @classmethod
  def sample(cls):
    return cls(np.random.randint(cls.MIN, cls.MAX + 1))

  def __index__(self):
    return self.value

  def __hash__(self):
    return hash(self.value)

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

  def __mod__(self, other):
    if not isinstance(other, Integer):
      raise ValueError('Operator % is not supported between Integer and {}'.format(other.__class__.__name__))
    return Integer(self.get_value() % other.get_value())

  def __eq__(self, other):
    if isinstance(other, int):
      return self.get_value() == other
    elif isinstance(other, Integer):
      return self.get_value() == other.get_value()
    elif isinstance(other, Value):
      return False
    else:
      raise ValueError('Operator == is not supported between Integer and {}'.format(other.__class__.__name__))

  def __ne__(self, other):
    if isinstance(other, int):
      return self.get_value() != other
    elif isinstance(other, Integer):
      return self.get_value() != other.get_value()
    elif isinstance(other, Value):
      return True
    else:
      raise ValueError('Operator != is not supported between Integer and {}'.format(other.__class__.__name__))

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
# ZERO = Integer(0)
# ONE = Integer(1)
# TWO = Integer(2)
# THREE = Integer(3)
# FOUR = Integer(4)
