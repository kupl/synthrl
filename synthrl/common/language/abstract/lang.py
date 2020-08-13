from abc import ABC
from abc import abstractmethod

class SyntaxError(Exception):
  pass

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

  @abstractmethod
  @property
  def production_space(self):
    pass

  @abstractmethod
  def product(self, token):
    pass

  @abstractmethod
  @classmethod
  def parse(cls, pgm):
    pass
