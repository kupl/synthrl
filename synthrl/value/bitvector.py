import numpy as np

from synthrl.utils.decoratorutils import classproperty
from synthrl.value.value import Value

class BitVector(Value):
  TYPE = np.dtype

  def __init__(self, value=0):
    if isinstance(value, BitVector):
      value = value.get_value()
    if isinstance(value, str):
      value = int(value)
    elif not (isinstance(value, int) or isinstance(value, np.integer)):
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

  @property
  def bits(self):
    return np.binary_repr(self.value, width=self.value.dtype.itemsize * 8)

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
    if not isinstance(n, BitVector):
      raise ValueError('{} is not a value of synthrl.value.BitVector.'.format(n))
    shifted = np.left_shift(self.value, n.get_value(),dtype=self.TYPE)
    return self.__class__(shifted)

  def __rshift__(self, n): #signed rshfit
    if not isinstance(n, BitVector):
      raise ValueError('{} is not a value of synthrl.value.BitVector.'.format(n))
    shifted=np.right_shift(self.value, n.get_value(),dtype=self.TYPE)
    return self.__class__(shifted)

  def uns_rshift(self, n):
    if not isinstance(n, BitVector):
      raise ValueError('{} is not a value of synthrl.value.BitVector.'.format(n))
    uns_self = self.unsigned
    uns_n = n.unsigned
    uns_shfited = np.right_shift(uns_self, uns_n).view(self.TYPE)
    return self.__class__(uns_shfited)

  def __eq__(self, other):
    if not isinstance(other, self.__class__):
      raise ValueError('Cannot compare {} and {}'.format(self.__class__.__name__, other.__class__.__name__))
    return (self.value - other.value) == 0
  
  def logical_neg(self):
    switched_bits = ['1' if x == '0' else '0' for x in self.bits]
    switched_bits.reverse()
    val = 0
    for i,x in enumerate(switched_bits[:-1]):
      val += (2**i)*int(x)
    val += - int(switched_bits[-1]) * (2**(len(switched_bits)-1))
    return self.__class__(val)

  def __ne__(self, other):
    return not (self == other)

  def __truediv__(self, other):
    if not isinstance(other, self.__class__):
      raise ValueError('Cannot compare {} and {}'.format(self.__class__.__name__, other.__class__.__name__))
    return self.__class__(self.value // other.value)
    
  def __mod__(self, other):
    if not isinstance(other, self.__class__):
      raise ValueError('Cannot compare {} and {}'.format(self.__class__.__name__, other.__class__.__name__))
    return self.__class__(self.value % other.value)

class BitVector16(BitVector):
  TYPE = np.int16
  
class BitVector32(BitVector):
  TYPE = np.int32

class BitVector64(BitVector):
  TYPE = np.int64
 
