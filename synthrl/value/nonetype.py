from synthrl.value.value import Value

class NoneType(Value):
  def __init__(self):
    pass

  def get_value(self):
    return None

  @classmethod
  def sample(cls):
    return None

  def __hash__(self):
    return hash(None)

  def __eq__(self, other):
    return other is None

  def __ne__(self, other):
    return other is not None

NONE = NoneType()