from copy import deepcopy

# Exception to handle when semantic is not defined
class UndefinedSemantics(Exception):
  def __init__(self, *args, **kwargs):
    super(UndefinedSemantics, self).__init__(*args, **kwargs)

# Abstract class that all DSLs must inherit
class Tree:
  def __init__(self, data='hole', children={}, parent=None):
    self.data = data
    self.children = children
    self.parent = parent

  def production_space(self, *args, **kwargs):
    # returns a tuple of target hole node and possible rule to fill the hole
    raise NotImplementedError

  def production(self, rule=None):
    # gets production rule from production space and apply to program tree
    raise NotImplementedError

  def interprete(self, *args, **kwargs):
    # gets needed information
    # and returns executed result of program
    raise NotImplementedError

  def pretty_print(self, file=None, *args, **kwargs):
    # print the program tree into program
    raise NotImplementedError

  @property
  def tokens(self):
    raise NotImplementedError

  def is_hole(self):
    return self.data == 'hole'

  def copy(self):
    node = self.__class__(data=self.data)
    children = {}
    for key, child in self.children.items():
      child = child.copy()
      child.parent = node
      children[key] = child
    node.children = children
    return node

  def __str__(self):
    return '{}({}, {})'.format(self.__class__.__name__, self.data, self.children)

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    if self.data != other.data:
      return False
    for key_self, key_other in zip(self.children.keys(), other.children.keys()):
      c_self = self.children[key_self]
      c_other = other.children[key_other]
      if c_self != c_other:
        return False
    return True

  def __ne__(self, other):
    return not (self == other)
