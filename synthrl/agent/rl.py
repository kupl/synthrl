import torch

from synthrl.agent.agent import Agent

class RLAgent(Agent):
  def __init__(self):
    self.policy_net = None
    self.value_net = None

  def take(self, action_space=[]):
    pass
