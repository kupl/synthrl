from abc import ABC
from abc import abstractclassmethod

class Testing(ABC):

  def __init__(self, language):
    self.type = language.VALUE
    self.n_input = language.N_INPUT
  
  def __call__(self, pgm1, pgm2):
    return self.testing(pgm1, pgm2)

  @abstractclassmethod
  def testing(self, pgm1, pgm2):
    pass
