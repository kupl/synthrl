from abc import ABC
from abc import abstractmethod

class Function(ABC):

  def __init__(self, language):
    self.language = language
    self.tokens = sorted(map(str, self.language.TOKENS))
    self.indices = {token: i for i, token in enumerate(self.tokens)}
  @abstractmethod
  def evaluate(self, state, **info):
    pass

  def __call__(self, state, **info):
    return self.evaluate(states=state, **info)

  def policy(self, state, **info):
    return self.evaluate[0]

  def value(self, state, **info):
    return self.evaluate(state, **info)[1]
