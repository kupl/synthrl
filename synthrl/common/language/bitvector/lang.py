from synthrl.common.language.abstract.exception import WrongProductionException
from synthrl.common.language.abstract.lang import Program

class BitVectorLang(Program):

  def __init__(self):
    self.start_node = ExprNode()
    self.node = None
    self.possible_actions = []

  @property
  def production_space(self):
    self.node, self.possible_actions = self.start_node.production_space()

  def product(self, action):
    if action not in self.possible_actions:
      raise WrongProductionException
