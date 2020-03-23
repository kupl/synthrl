from datetime import datetime
from datetime import timedelta

def parse_time(time='0h-0m-0s'):
  parsed = {'hours': 0, 'minutes': 0, 'seconds': 0}
  for t in time.split('-'):
    if t.endswith('h'):
      parsed['hours'] = int(t[:-1])
    elif t.endswith('m'):
      parsed['minutes'] = int(t[:-1])
    elif t.endswith('s'):
      parsed['seconds'] = int(t[:-1])
    else:
      raise ValueError('Unknown indecator: {}.'.format(t[-1]))
  return parsed

class Timer:
  def __init__(self, budget='0s'):
    self.budget = timedelta(**parse_time(budget))
    self.start_time = datetime.now()

  def __iter__(self):
    while True:
      delta = datetime.now() - self.start_time
      if delta > self.budget:
        break
      yield delta
    return delta

  def timeout(self):
    delta = datetime.now() - self.start_time
    if delta > self.budget:
      return True
    else:
      return False
