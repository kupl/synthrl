import logging
logging.basicConfig(level=logging.DEBUG)

from synthrl.agent import ExhaustiveAgent
from synthrl.example.synthesize import synthesize_from_oracle
from synthrl.language import ListLanguage
from synthrl.testing import random_testing
from synthrl.value import Integer
from synthrl.value import IntList
from synthrl.value.integer import ONE

oracle = lambda l, i: sorted(l)[i - ONE]
ioset = [
  ((IntList([1, 2, 3]), Integer(3)), Integer(3)),
  ((IntList([5, 1, 2]), Integer(3)), Integer(5))
]

program = synthesize_from_oracle(
  dsl=ListLanguage,
  dsl_opt={'input_types': (IntList, Integer), 'output_type': Integer},
  synthesizer=ExhaustiveAgent(mode='s'), 
  verifier=ExhaustiveAgent(mode='v'), 
  oracle=oracle, 
  ioset=ioset, 
  budget='1m', 
  testing=random_testing, 
  testing_opt={'input_types': (IntList, Integer), 'budget': '10s'}
)
program.pretty_print()
