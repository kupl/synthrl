import numpy as np
import sys

from synthrl.common.environment.dataset import Dataset
from synthrl.common.language.bitvector.lang import BitVectorLang


GREEN = '\033[92m'
RED = '\033[91m'
ENDC = '\033[0m'


def dfs(pgm, move, max_move, ioset):

  if move > max_move:
    return None, move

  if len(pgm) > 3:
    return None, move

  space = pgm.production_space
  if len(space) == 0:
    for i, o in ioset:
      out = pgm(i)
      # pylint: disable=too-many-function-args
      if out != BitVectorLang.VALUE(o):
        # Failed
        pgm = BitVectorLang()
        break
    else:
      # Pass all testcase.
      return pgm, move

    return None, move

  for action in np.random.permutation(space):
    copied = pgm.copy()
    print("To Copy")
    pgm.pretty_print()
    print("Copied")
    copied.pretty_print()
    
    copied.product(action)
    if action not in ["bop", "var","const"]:
      move += 1 
    copied, move = dfs(copied, move, max_move, ioset)
    if copied:
      return copied, move

  return None, move
  

def main(argv):
  if len(argv) < 3:
    print('Usage: python3 naive_random_script.py <dataset.json> <max_move>')
    sys.exit(1)
  dataset = Dataset.from_json(argv[1])

  success = 0
  failed = 0

  for element in dataset:
    oracle = element.oracle
    ioset = element.ioset

    max_move = int(argv[2])
    pgm, _ = dfs(BitVectorLang(), 0, max_move, ioset)
    found = pgm is not None
    
    if found:
      success += 1
    else:
      failed += 1
      pgm = BitVectorLang()
    print('-- oracle --')
    print(len(oracle))
    oracle.pretty_print()
    print('-- program --')
    if not pgm:
      pgm = BitVectorLang()
    pgm.pretty_print()
    print('------------')
    print(f'Result: {f"{GREEN}success{ENDC}" if found else f"{RED}failed {ENDC}"}')
    print('------------')
    print()

  print(f'{GREEN}Success:{ENDC} {success}')
  print(f'{RED}Failed:{ENDC} {failed}')


if __name__ == "__main__":
  main(sys.argv)
  