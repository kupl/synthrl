from synthrl.utils.decoratorutils import classproperty

# Exception to handle when invalid syntax is given
class SyntaxError(Exception):

  def __init__(self, *args, **kwargs):
    super(SyntaxError, self).__init__(*args, **kwargs)

# Exception to handle when semantic is not defined
class UndefinedSemantics(Exception):

  def __init__(self, *args, **kwargs):
    super(UndefinedSemantics, self).__init__(*args, **kwargs)

# Exception to handle when the unexpected behavior of program is observed
class UnexpectedException(Exception):

  def __init__(self, *args, **kwargs):
    super(UnexpectedException, self).__init__(*args, **kwargs)
  
# Exception to handle wrong production rule
class WrongProductionException(Exception):
  
  def __init__(self, *args, **kwargs):
    super(WrongProductionException, self).__init__(*args, **kwargs)

# Abstract class that very first symbol which represent root node should implement
class Tree:

  __slots__ = ['hole']

  def __init__(self, *args, **kwargs):
    self.hole = None

  @property
  def production_space(self):
    # returns a list of production space
    raise NotImplementedError

  def production(self, rule):
    # apply the production rule to hole node
    self.hole.production(rule)

  def interprete(self, inputs):
    # gets inputs
    # returns an executed result of program
    raise NotImplementedError

  def pretty_print(self, file=None):
    # print a program
    raise NotImplementedError

  @classproperty
  @classmethod
  def tokens(cls):
    # returns a list of all tokens
    raise NotImplementedError

  @classmethod
  def parse(cls, program):
    # gets program as string
    # returns a parsed Tree object
    raise NotImplementedError

  def __call__(self, *args, **kwargs):
    return self.interprete(*args, **kwargs)

# Abstract class that each non-terminal symbols should implement
class Node:

  __slots__ = [
    'children',
    'data',
    'parent',
  ]

  def __init__(self, data='HOLE', children={}, parent=None):
    self.data = data
    self.children = children
    self.parent = parent
    
  def production_space(self, *args, **kwargs):
    # returns a tuple of a hole node and a list of possible production rules
    raise NotImplementedError

  def production(self, rule):
    # gets one production rule to apply
    # returns self
    raise NotImplementedError

  def interprete(self, *args, **kwargs):
    # gets needed information
    # returns an executed result of node
    raise NotImplementedError

  def pretty_print(self, file=None, *args, **kwargs):
    # print a tree node
    raise NotImplementedError

  @classproperty
  @classmethod
  def tokens(cls):
    # returns a list of tokens
    raise NotImplementedError
