import numpy as np

from synthrl.agent.agent import Agent

class RandomAgent(Agent):
  def __init__(self):
    pass

  def take(self, state=None, action_space=[]):
    return np.random.choice(action_space)

  def reset(self):
    pass
