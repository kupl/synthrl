from synthrl.utils import Timer

def random_testing(pgm1=None, pgm2=None, input_types=None, budget='1s'):
  for _ in Timer(budget):
    inputs = [ty.sample() for ty in input_types]
    try:
      if pgm1.interprete(inputs) != pgm2.interprete(inputs):
        return inputs
    except Exception:
      pass
  return None
