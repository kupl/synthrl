from synthrl.common.verifier.exception import VerificationFailed
from synthrl.common.verifier.utils import check
from synthrl.common.verifier.verifier import Verifier

class SimpleVerifier(Verifier):

  def verify(self, candidate, ioset, max_move):

    pgm = None
    move = 0
    while move < max_move:
      if not pgm:
        pgm = self.language()

      space = pgm.production_space
      if len(space) == 0:
        dist_input = check(candidate, pgm, ioset, self.testing)
        if dist_input:
          break
        else:
          pgm = None

      else:
        policy = self.function.policy((pgm, candidate, ioset))
        action = self.function.sample(space, policy)
        pgm.product(action)
        move += 1

    else:
      raise VerificationFailed()

    return dist_input
