from synthrl.common.synthesizer.exception import SynthesisFailed
from synthrl.common.synthesizer.synthesizer import Synthesizer

class SimpleSynthesizer(Synthesizer):

  def __init__(self, language, function):
    super(SimpleSynthesizer, self).__init__(language, function)

  def synthesize(self, ioset, max_move):
    
    pgm = None
    for _ in range(max_move):
      if not pgm:
        pgm = self.language()
      
    else:
      raise SynthesisFailed()
      

    
    return pgm