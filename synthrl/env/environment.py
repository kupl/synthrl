
# Abstract class that all environment classes should inherit
class Environment:
  def __init__(self):
    raise NotImplementedError

  def step(self, action=None):
    raise NotImplementedError
  
  def rest(self):
    raise NotImplementedError
