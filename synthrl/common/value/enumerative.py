from abc import abstractmethod

from synthrl.common.utils import classproperty
from synthrl.common.value.value import Value

class Enumerative(Value):

  @classproperty
  @classmethod
  @abstractmethod
  def MAX_LENGTH(cls):
    pass

  def __iter__(self):
    for v in self.value:
      yield v
    