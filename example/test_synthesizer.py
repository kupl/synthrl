from synthrl.agent import RandomAgent
from synthrl.language import ListLanguage
from synthrl.wrapper import Synthesizer

oracle = lambda l, i: sorted(l)[i - 1]
ioset = [
  (([1, 2, 3], 3), 3),
  (([5, 1, 2], 3), 5)
]

synthesizer = Synthesizer(dsl=ListLanguage, synth=RandomAgent(), alt=RandomAgent())
synthesizer.synthesize_from_oracle(io=ioset, oracle=oracle)
