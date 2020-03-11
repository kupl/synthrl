from copy import deepcopy

# Abstract class that all DSLs must inherit
class Tree:
  def __init__(self, data='hole', children={}, parent=None):
    self.data = data
    self.children = children
    self.parent = parent

  def production_space(self):
    # returns a tuple of target hole node and possible rule to fill the hole
    raise NotImplementedError

  def production(self, rule=None):
    # gets production rule from production space and apply to program tree
    raise NotImplementedError

  def interprete(self, *args, **kwargs):
    # gets needed information
    # and returns executed result of program
    raise NotImplementedError

  def pretty_print(self, *args, **kwargs):
    # print the program tree into program
    raise NotImplementedError

  def copy(self):
    return deepcopy(self)

  def __str__(self):
    return '{}({}, {})'.format(self.__class__.__name__, self.data, self.children)

  def __repr__(self):
    return str(self)
