from abc import ABC
from abc import abstractmethod

class Synthesizer(ABC):

  def __init__(self, language, function):
    self.language = language
    self.function = function

  @abstractmethod
  def synthesize(self, ioset):
    pass
