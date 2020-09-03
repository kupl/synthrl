import numpy as np
import sys

from synthrl.utils.trainutils import Dataset
from synthrl.language.bitvector.lang import BitVectorLang

GREEN = '\033[92m'
RED = '\033[91m'
ENDC = '\033[0m'

def main(argv):
  if len(argv) < 3:
    print('Usage: python3 naive_random_script.py <dataset.json> <max_move>')
    sys.exit(1)
  dataset = Dataset.from_json(argv[1])

  success = 0
  failed = 0

  for element in dataset:
    oracle = element.program
    ioset = element.ioset

    pgm = None
    space = None
    found = False
    for _ in range(int(argv[2])):

      # Create new tree
      if not pgm:
        pgm = BitVectorLang()
        space = pgm.production_space()

      # Take action
      action = np.random.choice(space)
      pgm.production(action)

      # Check if complete
      space = pgm.production_space()
      if len(space) == 0:
        for i, o in ioset:
          out = pgm(i)
          # pylint: disable=too-many-function-args
          if out != BitVectorLang.BitVector(o):
            # Failed
            pgm = None
            break
        else:
          found = True
      
      # If found
      if found:
        break
        
    if found:
      success += 1
    else:
      failed += 1
    print('-- oracle --')
    oracle.pretty_print()
    print('-- program --')
    if not pgm:
      pgm = BitVectorLang()
    pgm.pretty_print()
    print('------------')
    print(f'Result: {f"{GREEN}success{ENDC}" if found else f"{RED}failed{ENDC}"}')
    print('------------')
    print()

  print(f'{GREEN}Success:{ENDC} {success}')
  print(f'{RED}Failed:{ENDC} {failed}')

if __name__ == "__main__":
  main(sys.argv)