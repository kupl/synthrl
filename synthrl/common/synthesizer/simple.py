from synthrl.common.synthesizer.exception import SynthesisFailed
from synthrl.common.synthesizer.synthesizer import Synthesizer
from synthrl.common.synthesizer.utils import check

class SimpleSynthesizer(Synthesizer):

  def synthesize(self, ioset, max_move):
    
    pgm = None
    move = 0
    while move < max_move:
      if not pgm:
        pgm = self.language()

      space = pgm.production_space
      if len(space) == 0:
        if check(pgm, ioset):
          break
        else:
          pgm = None
          
      else:
        policy = self.function.policy((pgm, ioset))
        action = self.function.sample(space, policy)
        pgm.product(action)
        move += 1

    else:
      raise SynthesisFailed()
          
    return pgm
