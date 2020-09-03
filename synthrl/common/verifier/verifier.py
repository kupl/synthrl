from abc import ABC
from abc import abstractclassmethod

class Verifier(ABC):

  def __init__(self, language, function, testing):
    self.language = language
    self.function = function
    self.testing = testing

  @abstractclassmethod
  def verify(self, program, ioset):
    pass
