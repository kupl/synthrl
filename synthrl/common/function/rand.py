import numpy as np

from synthrl.common.function.function import Function
from synthrl.common.utils.mathutils import normalize

class RandomFunction(Function):

  def evaluate(self, state, **info):
    policy = np.random.uniform(size=len(self.tokens))
    policy = policy / policy.sum()
    value = np.random.uniform()
    return policy, value
