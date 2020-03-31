from synthrl.env.environment import Environment
from synthrl.language import UndefinedSemantics
from synthrl.utils import IOSet

class ComparisonFailed(Exception):
  def __init__(self, *args, **kwargs):
    super(ComparisonFailed, self).__init__(*args, **kwargs)

class DistinguishingNotFound(Exception):
  def __init__(self, *args, **kwargs):
    super(DistinguishingNotFound, self).__init__(*args, **kwargs)

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
        try:
          for i, o in self.ioset:
            result = self.candidate.interprete(i)
            if o != result:
              raise ComparisonFailed()
          self.candidate_reward = 1
          self.candidate_terminate = True
          ## logging ##
          print('--candidate--')
          self.candidate.pretty_print()
          ## logging ##
        except (UndefinedSemantics, ComparisonFailed):
          self.candidate_reward = -1
          self.candidate = self.dsl()
          self.node, self.space = self.candidate.production_space()
          return
    if not self.candidate_terminate:
      return

    if not self.alternative_terminate:
      self.node, self.space = self.alternative.production_space()
      if len(self.space) > 0:
        self.alternative_reward = 0
        return
      else:
        try:
          for i, o in self.ioset:
            result = self.alternative.interprete(i)
            if o != result:
              raise ComparisonFailed()
          ## logging ##
          print('--alternative--')
          self.alternative.pretty_print()
          ## logging ##
          self.distinguishing_input = self.testing(self.candidate, self.alternative)
          if self.distinguishing_input is None:
            raise DistinguishingNotFound()
          self.alternative_reward = 1
          self.alternative_terminate = True
        except (UndefinedSemantics, ComparisonFailed, DistinguishingNotFound):
          self.alternative_reward = -1
          self.alternative = self.dsl()
          self.node, self.space = self.alternative.production_space()
          return
    if not self.alternative_terminate:
      return

    self.alternative_terminate = True
    self.node, self.space = None, []
