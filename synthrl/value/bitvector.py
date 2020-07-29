import numpy as np

from synthrl.utils.decoratorutils import classproperty
from synthrl.value.integer import Integer
from synthrl.value.value import Value

class BitVector(Value):
  TYPE = np.dtype

  def __init__(self, value=0):
    if isinstance(value, BitVector):
      value = value.get_value()
    if isinstance(value, str):
      value = int(value)
    elif not (isinstance(value, int) or isinstance(value, self.TYPE)):
      raise ValueError('{} is not an integer.'.format(value))
    self.value = self.TYPE(value)

  def get_value(self):
    return int(self.value)

  @classmethod
  def sample(cls):
    iinfo = np.iinfo(cls.TYPE)
    return cls(np.random.randint(iinfo.min, iinfo.max + 1, dtype=cls.TYPE))

  @property
  def unsigned(self):
    utype = np.dtype('uint{}'.format(self.value.dtype.itemsize * 8))
    return self.value.view(utype)

  @classproperty
  @classmethod
  def size(cls):
    iinfo = np.iinfo(cls.TYPE)
    return iinfo.max - iinfo.min + 1

  def __neg__(self):
    return self.__class__(-self.value)

  def __add__(self, other):
    if not isinstance(other, self.__class__):
      raise ValueError('Operator + is not supported between {} and {}'.format(self.__class__.__name__, other.__class__.__name__))
    return self.__class__(self.value + other.value)

  def __sub__(self, other):
    if not isinstance(other, self.__class__):
      raise ValueError('Operator - is not supported between {} and {}'.format(self.__class__.__name__, other.__class__.__name__))
    return self.__class__(self.value - other.value)

  def __mul__(self, other):
    if not isinstance(other, self.__class__):
      raise ValueError('Operator * is not supported between {} and {}'.format(self.__class__.__name__, other.__class__.__name__))
    return self.__class__(self.value * other.value)

  def __and__(self, other):
    if not isinstance(other, self.__class__):
      raise ValueError('Operator & is not supported between {} and {}'.format(self.__class__.__name__, other.__class__.__name__))
    return self.__class__(np.bitwise_and(self.value, other.value))
  
  def __or__(self, other):
    if not isinstance(other, self.__class__):
      raise ValueError('Operator | is not supported between {} and {}'.format(self.__class__.__name__, other.__class__.__name__))
    return self.__class__(np.bitwise_or(self.value, other.value))

  def __xor__(self, other):
    if not isinstance(other, self.__class__):
      raise ValueError('Operator ^ is not supported between {} and {}'.format(self.__class__.__name__, other.__class__.__name__))
    return self.__class__(np.bitwise_xor(self.value, other.value))

  def __lshift__(self, n):
    if not isinstance(Integer):
      raise ValueError('{} is not a value of synthrl.value.Integer.'.format(n))
    return self.__class__(np.left_shift(self.value, n.get_value()))

  def __rshift__(self, n):
    if not isinstance(Integer):
      raise ValueError('{} is not a value of synthrl.value.Integer.'.format(n))
    return self.__class__(np.right_shift(self.value, n.get_value()))

  def __eq__(self, other):
    if not isinstance(other, self.__class__):
      raise ValueError('Cannot compare {} and {}'.format(self.__class__.__name__, other.__class__.__name__))
    return (self.value - other.value) == 0

  def __ne__(self, other):
    return not (self == other)

class BitVector16(BitVector):
  TYPE = np.int16
  
class BitVector32(BitVector):
  TYPE = np.int32

class BitVector64(BitVector):
  TYPE = np.int64
