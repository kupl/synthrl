
# Abstract class that all value classes should inherit
class Value:
  def __init__(self, *args, **kwargs):
    # check the condition of the value
    # must implement the constructor that gets the object of itself
    raise NotImplementedError

  def get_value(self):
    # returns the actual value
    raise NotImplementedError

  @classmethod
  def sample(cls):
    # randomly generate value
    raise NotImplementedError

  def copy(self):
    return self.__class__(self)

  def __str__(self):
    return self.__repr__()

  def __call__(self):
    return self.get_value()

  def __repr__(self):
    return str(self.get_value())
