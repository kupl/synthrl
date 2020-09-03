import numpy as np
import sys
import torch 
from synthrl.common.environment.dataset import Dataset
from synthrl.common.language.bitvector.lang import BitVectorLang
from synthrl.common.language.bitvector.lang import ExprNode
from synthrl.common.language.bitvector.lang import BOPNode
from synthrl.common.language.bitvector.lang import ConstNode
from synthrl.common.language.bitvector.lang import ParamNode


from synthrl.common.value.bitvector import BitVector
from synthrl.common.value.bitvector import BitVector16
from synthrl.common.value.bitvector import BitVector32
from synthrl.common.value.bitvector import BitVector64
from synthrl.common.function.rnn import RNNFunction




GREEN = '\033[92m'
RED = '\033[91m'
ENDC = '\033[0m'


def load_poilcy(pgm, ioset, agent):
  tokens = sorted(BitVectorLang.TOKENS)
  states = [(pgm,ioset)]
  policy, value = agent.evaluate(states=states)
  policy =policy.tolist()[0]
  assert len(BitVectorLang.TOKENS) == len(policy)
  token_by_policy = list(zip(tokens, policy))
  sorted_tbp  = sorted(token_by_policy, key=lambda x: x[1], reverse=True)
  prioritized_space = [x[0]  for x  in sorted_tbp] 
  return prioritized_space


def dfs(pgm, move, max_move, ioset, agent):

  if move > max_move:
    return None, move

  if len(pgm) > 3:
    return None, move
  # space = pgm.production_space()
  # if len(space) == 0:
  if pgm.is_complete():
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

  prioritized_space = load_poilcy(pgm, ioset, agent)


  for action in prioritized_space:
    copied = pgm.copy()
    if action in ExprNode.TOKENS: #ExprNode.tokens=["neg","arith-neg"], due to way of construction
      copied.product(action)
    elif action in BOPNode.TOKENS:
      copied.product("bop")
      copied.product(action)
    elif action in ConstNode.TOKENS:
      copied.product("const")
      copied.product(int(action))
    elif action in ParamNode.TOKENS:
      copied.product("var")
      copied.product(action)
    if action not in ["bop", "var","const"]:
      move += 1 
    copied, move = dfs(copied, move, max_move, ioset, agent)    
    if copied:
      return copied, move

  return None, move
  

def main(argv):
  if len(argv) < 3:
    print('Usage: python3 naive_random_script.py <dataset.json> <max_move>')
    sys.exit(1)
  dataset = Dataset.from_json(argv[1])

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  agent = RNNFunction.load(argv[3],device)

  agent.token_emb.eval()
  agent.value_emb.eval()
  agent.network.eval()

  success = 0
  failed = 0
  
  for element in dataset:
    oracle = element.oracle
    ioset = element.ioset
    
    max_move = int(argv[2])
    pgm, _ = dfs(BitVectorLang(), 0, max_move, ioset, agent)
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
  