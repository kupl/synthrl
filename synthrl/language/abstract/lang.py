from synthrl.utils.decoratorutils import classproperty

# Exception to handle when semantic is not defined
class UndefinedSemantics(Exception):
  def __init__(self, *args, **kwargs):
    super(UndefinedSemantics, self).__init__(*args, **kwargs)

# Abstract class that very first symbol which represent root node should implement
class Tree:

  __slots__ = ['hole']

  def __init__(self, *args, **kwargs):
    self.hole = None

  @property
  def production_space(self):
    raise NotImplementedError

  def production(self, rule):
    raise NotImplementedError

  def interprete(self, inputs):
    raise NotImplementedError

  def pretty_print(self, file=None):
    raise NotImplementedError

  @classproperty
  @classmethod
  def tokens(cls):
    raise NotImplementedError

# Abstract class that each non-terminal symbols should implement
class Node:

  __slots__ = [
    'children',
    'data',
    'parent',
  ]

  def __init__(self, data='hole', children={}, parent=None):
    self.data = data
    self.children = children
    self.parent = parent
    
  def production_space(self, *args, **kwargs):
    # returns a list of possible production rules
    raise NotImplementedError

  def production(self, rule):
    # gets one production rule to apply
    # returns self
    raise NotImplementedError

  def interprete(self, *args, **kwargs):
    # gets needed information
    # returns an executed result of program
    raise NotImplementedError

  def pretty_print(self, file=None, *args, **kwargs):
    # print a tree node
    raise NotImplementedError

  @classproperty
  @classmethod
  def tokens(cls):
    # returns a list of tokens
    raise NotImplementedError
