from synthrl.env.environment import Environment
from synthrl.utils import IOSet

class MAEnvironment(Environment):
  def __init__(self, ioset=[], dsl=None, testing=None):
    self.ioset = IOSet(ioset)
    self.dsl = dsl
    self.testing = testing
    self.reset()

  def reset(self):
    self.candidate = self.dsl()
    self.alternative = self.dsl()
    self.candidate_reward = 0
    self.alternative_reward = 0
    self.candidate_terminate = False
    self.alternative_terminate = False
    self.update()
    return self.state, self.reward, self.termination

  @property
  def state(self):
    return self.ioset.copy(), self.candidate.copy(), self.alternative.copy()

  @property
  def action_space(self):
    return self.space

  @property
  def termination(self):
    return self.candidate_terminate, self.alternative_terminate

  @property
  def reward(self):
    return self.candidate_reward, self.alternative_reward

  @property
  def program(self):
    return self.candidate.copy()

  def step(self, action=None):
    if not action in self.space:
      raise ValueError('The given action({}) is not valid.'.format(action))
    self.node.production(action)
    self.update()
    return self.state, self.reward, self.termination

  def update(self):
    if not self.candidate_terminate:
      self.node, self.space = self.candidate.production_space()
      if len(self.space) > 0:
        self.candidate_reward = 0
        return
      else:
        for i, o in self.ioset:
          try:
            if o != self.candidate.interprete(i):
              self.candidate_reward = -1
              self.candidate = self.dsl()
              self.node, self.space = self.candidate.production_space()
              return
          except Exception:
            self.candidate_reward = -1
            self.candidate = self.dsl()
            self.node, self.space = self.candidate.production_space()
            return
        self.candidate_reward = 1
        self.candidate_terminate = True

    if not self.alternative_terminate:
      self.node, self.space = self.alternative.production_space()
      if len(self.space) > 0:
        self.alternative_reward = 0
        return
      else:
        for i, o in self.ioset:
          try:
            if o != self.alternative.interprete(i):
              self.alternative_reward = -1
              self.alternative = self.dsl()
              self.node, self.space = self.alternative.production_space()
              return
          except Exception:
            self.alternative_reward = -1
            self.alternative = self.dsl()
            self.node, self.space = self.alternative.production_space()
            return
        ## logging ##
        print('--alternative--')
        self.alternative.pretty_print()
        ## logging ##
        self.distinguishing_input = self.testing(self.candidate, self.alternative)
        if self.distinguishing_input is None:
          self.alternative_reward = -1
          self.alternative = self.dsl()
          self.node, self.space = self.alternative.production_space()
          return
        self.alternative_reward = 1
        self.alternative_terminate = True  

    self.alternative_terminate = True
    self.node, self.space = None, []
