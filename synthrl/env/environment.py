from synthrl.utils import TimeoutException

# Abstract class that all environment classes should inherit
class Environment:
  def __init__(self, *args, **kwargs):
    # gets needed objects to create environment
    raise NotImplementedError
  
  @property
  def state(self):
    # returns state of the environment
    raise NotImplementedError

  @property
  def action_space(self):
    # returns possible action space
    raise NotImplementedError

  def step(self, action=None):
    if self.timer.timeout:
      raise TimeoutException
    return self.apply_step(action=action)

  def apply_step(self, action=None):
    # gets an action from action space
    # and returns a tuple of new state, reward, and bool that represent if the state is terminal state
    raise NotImplementedError

  def reset(self):
    # reset all the variables to initial state
    # and returns a tuple of initial state, reward(0), and terminal(False)
    raise NotImplementedError

  @property
  def program(self):
    # returns the current program
    raise NotImplementedError

  def set_timer(self, timer=None):
    self.timer = timer
