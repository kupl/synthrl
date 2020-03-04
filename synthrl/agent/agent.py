
# Abstract class that all agent must inherit
class Agent:
  def __init__(self):
    # initialize required components
    raise NotImplementedError
    
  def take(self, action_space=[]):
    # takes action space
    # and returns the proper action
    raise NotImplementedError
