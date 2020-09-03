from synthrl.common.verifier.testing.testing import Testing

class RandomTesting(Testing):

  def __init__(self, language, max_attempt):
    super(RandomTesting, self).__init__(language)
    self.max_attempt = int(max_attempt)

  def testing(self, pgm1, pgm2):
    for _ in range(self.max_attempt):
      input = [self.type.sample() for _ in range(self.n_input)]
      if pgm1(input) != pgm2(input):
        return input
    return None