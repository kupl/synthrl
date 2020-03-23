from synthrl.agent import RandomAgent
from synthrl.example.synthesize import synthesize_from_oracle
from synthrl.language import ListLanguage
from synthrl.testing import random_testing
from synthrl.value import Integer
from synthrl.value import IntList

oracle = lambda l, i: sorted(l)[i - 1]
ioset = [
  (([1, 2, 3], 3), 3),
  (([5, 1, 2], 3), 5)
]

program = synthesize_from_oracle(dsl=ListLanguage, synthesizer=RandomAgent(), verifier=RandomAgent(), oracle=oracle, ioset=ioset, budget='1m', testing=random_testing, testing_opt={'input_types': (IntList, Integer), 'budget': '10s'})
program.pretty_print()
