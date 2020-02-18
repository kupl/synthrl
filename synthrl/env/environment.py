
# Abstract class that all environment classes should inherit
class Environment:
  def __init__(self):
    pass

  def step(self, action=None):
    pass
  
  def rest(self):
    pass
