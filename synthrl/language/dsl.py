
# Abstract class that all DSL must inherit
class Tree:
  def __init__(self, data='hole', children={}, parent=None):
    self.data = data
    self.children = children
    self.parent = parent

  def production_space(self):
    raise NotImplementedError

  def production(self, rule=None):
    raise NotImplementedError

  def interprete(self, *args, **kwargs):
    raise NotImplementedError

  def pretty_print(self, *args, **kwargs):
    raise NotImplementedError

  def __str__(self):
    return '{}({}, {})'.format(self.__class__.__name__, self.data, self.children)

  def __repr__(self):
    return str(self)
