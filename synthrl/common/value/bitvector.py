import numpy as np

from synthrl.common.value.value import Value
from synthrl.common.utils import classproperty

class BitVector(Value):

  TYPE = np.dtype

  @classproperty
  @classmethod
  def N_VALUE(cls):
    return cls.TYPE().itemsize * 8

  @classproperty
  @classmethod
  def N_BIT(cls):
    return np.iinfo(cls.TYPE).bits

  def __init__(self, value=0):
    if isinstance(value, BitVector):
      value = value.value
    elif isinstance(value, str):

      value = int(value, 16 if value.startswith('0x') else 10)
    elif not isinstance(value, (int, np.integer)):
      raise ValueError(f'{value} is not an integer.')
    self.value = self.TYPE(value)

  @classmethod
  def sample(cls):
    iinfo = np.iinfo(cls.TYPE)
    return cls(np.random.randint(iinfo.min, iinfo.max + 1, dtype=cls.TYPE))

  @property
  def unsigned(self):
    utype = np.dtype(f'uint{self.value.dtype.itemsize * 8}')
    return self.value.view(utype)

  @property
  def index(self):
    return self.unsigned

  @property
  def signed(self):
    return self.value

  def __int__(self):
    return int(self.value)

  def __repr__(self):
    return np.binary_repr(self.value, width=self.value.dtype.itemsize * 8)

  def __neg__(self):
    return self.__class__(-self.value)

  def __invert__(self):
    return self.__class__(np.bitwise_not(self.value))

  def __eq__(self, other):
    return self.value == other.value

  def __ne__(self, other):
    return self.value != other.value

  def __add__(self, other):
    if not isinstance(other, self.__class__):
      raise TypeError(f'Operator + is not supported between {self.__class__.__name__} and {other.__class__.__name__}.')
    return self.__class__(self.value + other.value)

  def __sub__(self, other):
    if not isinstance(other, self.__class__):
      raise TypeError(f'Operator - is not supported between {self.__class__.__name__} and {other.__class__.__name__}.')
    return self.__class__(self.value - other.value)

  def __mul__(self, other):
    if not isinstance(other, self.__class__):
      raise TypeError(f'Operator * is not supported between {self.__class__.__name__} and {other.__class__.__name__}.')
    return self.__class__(self.value * other.value)
  
  def __truediv__ (self, other):
    if not isinstance(other, self.__class__):
      raise TypeError(f'Operator // is not supported between {self.__class__.__name__} and {other.__class__.__name__}.')
    return self.__class__(self.value // other.value)

  def __mod__ (self, other):
    if not isinstance(other, self.__class__):
      raise TypeError(f'Operator % is not supported between {self.__class__.__name__} and {other.__class__.__name__}.')
    return self.__class__(self.value % other.value)

  def __or__(self, other):
    if not isinstance(other, self.__class__):
      raise TypeError(f'Operator | is not supported between {self.__class__.__name__} and {other.__class__.__name__}.')
    return self.__class__(np.bitwise_or(self.value, other.value))

  def __and__(self, other):
    if not isinstance(other, self.__class__):
      raise TypeError(f'Operator & is not supported between {self.__class__.__name__} and {other.__class__.__name__}.')
    return self.__class__(np.bitwise_and(self.value, other.value))

  def __xor__(self, other):
    if not isinstance(other, self.__class__):
      raise TypeError(f'Operator ^ is not supported between {self.__class__.__name__} and {other.__class__.__name__}.')
    return self.__class__(np.bitwise_xor(self.value, other.value))

  def __lshift__(self, other):
    if not isinstance(other, self.__class__):
      raise TypeError(f'Operator << is not supported between {self.__class__.__name__} and {other.__class__.__name__}.')
    return self.__class__(np.left_shift(self.value, other.value))
  
  def unsigned_rshift(self, other):
    if not isinstance(other, self.__class__):
      raise TypeError(f'"unsigned_rshift" is not supported between {self.__class__.__name__} and {other.__class__.__name__}.')
    return self.__class__(self.unsigned >> other.unsigned)

  def __rshift__(self, other):
    if not isinstance(other, self.__class__):
      raise TypeError(f'Operator >> is not supported between {self.__class__.__name__} and {other.__class__.__name__}.')
    return self.__class__(np.right_shift(self.value, other.value))

class BitVector8(BitVector):
  TYPE = np.int8

class BitVector16(BitVector):
  TYPE = np.int16

class BitVector32(BitVector):
  TYPE = np.int32

class BitVector64(BitVector):
  TYPE = np.int64
