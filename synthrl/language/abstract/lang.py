from synthrl.utils.decoratorutils import classproperty

# very outer class that represent root node
class Tree:
  def __init__(self, *args, **kwargs):
    pass

  def production_space(self):
    raise NotImplementedError

  def production(self, rule):
    raise NotImplementedError

  def pretty_print(self, file=None):
    raise NotImplementedError

  @classproperty
  @classmethod
  def tokens(cls):
    raise NotImplementedError

class Node:
  def __init__(self, data='hole', children={}, parent=None):
    self.data = data
    self.children = children
    self.parent = parent
    
  def production_space(self):
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
