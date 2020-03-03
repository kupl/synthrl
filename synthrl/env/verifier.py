from collections import defaultdict

from synthrl.env.environment import Environment

class VerifierEnvironment(Environment):
  def __init__(self, program=None, ioset=[], cand=None):
    self.orig_program = program.copy()
    self.ioset = ioset
    self.cand = cand.copy()

    self.reset()

  def reset(self):
    self.program = self.orig_program.copy()
    self.__update_space()
    return self.state, 0, False

  def __update_space(self):
    self.node, self.space = self.program.production_space()

  @property
  def state(self):
    return self.program.copy(), self.ioset, self.cand.copy()

  def action_space(self):
    return self.space

  def step(self, action=None):
    if action not in self.space:
      raise ValueError('Action({}) is not valid.'.format(action))
    self.node.production(action)
    self.__update_space()
    if len(self.space) == 0:
      for i, o in self.ioset:
        mem = defaultdict(list)
        mem['v0'] = i[0]
        mem['v1'] = i[1]
        if o != self.program.interprete(mem)['v19']:
          return self.state, 0, True
      return self.state, 1, True
    return self.state, 0, False
