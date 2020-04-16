from synthrl.value.value import Value

class NoneType(Value):
  def __init__(self):
    pass

  def get_value(self):
    return None

  @classmethod
  def sample(cls):
    return None

  def __eq__(self, other):
    return not other.get_value()

  def __ne__(self, other):
    return not (self == other)

NONE = NoneType()