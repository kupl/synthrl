from synthrl.utils.decoratorutils import classproperty

class Tree:
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

  @property
  def spec(self):
    # returns a dictionary that contains all information to create hole
    return {}
  