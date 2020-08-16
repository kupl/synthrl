from abc import ABC
from abc import abstractmethod

HOLE = 'HOLE'

class Program(ABC):

  def __init__(self):
    pass

  @abstractmethod
  def interprete(self, input):
    pass

  @abstractmethod
  def pretty_print(self, file=None):
    pass

  def __call__(self, *args, **kwargs):
    return self.interprete(*args, **kwargs)

  @property
  @abstractmethod
  def production_space(self):
    pass

  @abstractmethod
  def product(self, token):
    pass

  @classmethod
  @abstractmethod
  def parse(cls, pgm):
    pass

  @abstractmethod
  def copy(self):
    pass

class Tree:

  def __init__(self, data=HOLE, children=None):
    self.data = data
    self.children = children if children else {}

  def copy(self):
    copied_children = {key: child.copy() for key, child in self.children.items()}
    copied = self.__class__(self.data, copied_children)
    return copied
