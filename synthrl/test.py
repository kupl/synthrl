from synthrl.utils import Timer

def random_testing(pgm1=None, pgm2=None, input_types=None, budget='1s'):
  for t in Timer(budget):
    inputs = [ty.sample() for ty in input_types]
    if pgm1.interprete(inputs) != pgm2.interprete(inputs):
      return inputs
  return None
