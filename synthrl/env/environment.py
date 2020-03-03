
# Abstract class that all environment classes should inherit
class Environment:
  def __init__(self, *args, **kwargs):
    # gets needed objects to create environment
    raise NotImplementedError
  
  def action_space(self):
    # returns possible action space
    raise NotImplementedError

  def step(self, action=None):
    # gets an action from action space
    # and returns a tuple of new state, reward, and bool that represent if the state is terminal state
    raise NotImplementedError

  def reset(self):
    # reset all the variables to initial state
    # and returns a tuple of initial state, reward(0), and terminal(False)
    raise NotImplementedError
