from abc import ABC
from abc import abstractmethod
import numpy as np

from synthrl.common.utils import normalize

class Function(ABC):

  def __init__(self, language):
    self.language = language
    self.tokens = sorted(map(str, self.language.TOKENS))
    self.indices = {token: i for i, token in enumerate(self.tokens)}

  @abstractmethod
  def evaluate(self, state, **info):
    pass

  def __call__(self, state, **info):
    return self.evaluate(state=state, **info)

  def policy(self, state, **info):
    return self.evaluate(state, **info)[0]

  def value(self, state, **info):
    return self.evaluate(state, **info)[1]

  def sample(self, space, policy):
    space = [self.indices[action] for action in space]
    policy = policy[space]
    selected = np.random.choice(space, p=normalize(policy))
    return self.tokens[selected]