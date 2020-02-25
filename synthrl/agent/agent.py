
# Abstract class that all agent must inherit
class Agent:
  def __init__(self):
    raise NotImplementedError
    
  def take(self, action_space):
    raise NotImplementedError
