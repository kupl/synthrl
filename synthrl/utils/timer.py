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

def Timer(budget='0s'):
  budget = timedelta(**parse_time(budget))
  start_time = datetime.now()
  while True:
    delta = datetime.now() - start_time
    if delta > budget:
      break
    yield delta
  return delta
