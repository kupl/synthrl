from synthrl.env.environment import Environment
from synthrl.utils import IOSet

class MAEnvironment(Environment):
  def __init__(self, ioset=[], dsl=None, testing=None):
    self.ioset = IOSet(ioset)
    self.dsl = dsl
    self.testing = testing

  @property
  def state(self):
    return self.ioset.copy(), self.candidate.copy(), self.alternative.copy()

  def update_space(self):
    if not self.candidate_terminate:
      self.node, self.space = self.candidate.production_rule()
      if len(self.space) > 0:
        return
    self.candidate_terminate = True
    if not self.alternative_terminate:
      self.node, self.space = self.alternative.production_rule()
      if len(self.space) > 0:
        return
    self.alternative_terminate = True
    self.node, self.space = None, []

  @property
  def action_space(self):
    return self.space

  @property
  def termination(self):
    return self.candidate_terminate, self.alternative_terminate

  def candidate_reward(self):
    if not self.candidate_terminate:
      return 0
    for i, o in self.ioset:
      if o != self.candidate.interprete(i):
        return -1
    return 1

  def alternative_reward(self):
    if not self.alternative_terminate:
      return 0
    for i, o in self.ioset:
      if o != self.alternative.interprete(i):
        return -1
    self.distingusing_input = self.testing(self.candidate, self.alternative)
    if self.distingusing_input is None:
      return -1
    return 1

  @property
  def reward(self):
    return self.candidate_reward, self.alternative_reward

  def step(self, action=None):
    if not action in self.space:
      raise ValueError('The given action({}) is invalid.'.format(action))
    self.node.production(action)
    self.update_space()
    return self.state, self.reward, self.termination

  def reset(self):
    self.candidate = self.dsl()
    self.alternative = self.dsl()
    self.candidate_terminate = False
    self.alternative_terminate = False
    self.update_space()
    return self.state, self.reward, self.termination
