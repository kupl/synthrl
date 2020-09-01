#    P -> Expr
# Expr -> Var
#       | Cnst
#       | Bop
#       | - Expr  (arithmetic negation)
#       | ~ Expr  (bitwise NOT)
#       | if Bool then Expr else Expr
#  Bop -> Expr + Expr
#       | Expr - Expr
#       | Expr * Expr
#       | Expr / Expr
#       | Expr % Expr
#       | Expr || Expr
#       | Expr & Expr
#       | Expr ^ Expr
#       | Expr >>_s Expr
#       | Expr >>_u Expr
#       | Expr << Expr
# Bool -> true
#       | false
#       | Expr = Expr
#       | Bool and Bool
#       | Bool or Bool
#       | not Bool
# Cnst -> 0x00
#       | 0x01
#       | 0x02
#       | 0x03
#       | 0x04
#       | 0x05
#       | 0x06
#       | 0x07
#       | 0x08
#       | 0x09
#       | 0x0A
#       | 0x0B
#       | 0x0C
#       | 0x0D
#       | 0x0E
#       | 0x0F
#       | 0x10
#  Var -> param0
#       | param1

from synthrl.common.language.abstract.exception import SyntaxError
from synthrl.common.language.abstract.exception import WrongProductionException
from synthrl.common.language.abstract.lang import Program
from synthrl.common.language.bitvector.grammar import BOPNode
from synthrl.common.language.bitvector.grammar import ConstNode
from synthrl.common.language.bitvector.grammar import ExprNode
from synthrl.common.language.bitvector.grammar import ParamNode
from synthrl.common.language.bitvector.grammar import parse
from synthrl.common.utils import classproperty
from synthrl.common.value.bitvector import BitVector
import synthrl.common.value.bitvector as bitvector


class BitVectorLang(Program):

  VECTOR_SIZE = 16
  __BitVector = None

  @classproperty
  @classmethod
  def VALUE(cls):
    return cls.BITVECTOR

  @classproperty
  @classmethod
  def N_INPUT(cls):
    return 2

  @classproperty
  @classmethod
  def BITVECTOR(cls):
    if not cls.__BitVector or cls.__BitVector.N_BIT != cls.VECTOR_SIZE:
      cls.__BitVector = getattr(bitvector, f'BitVector{cls.VECTOR_SIZE}')
    return cls.__BitVector

  def __init__(self, start_node=None):
    self.start_node = start_node if start_node else ExprNode()
    self.node = None
    self.possible_actions = []

  @property
  def production_space(self):
    self.node, self.possible_actions = self.start_node.production_space()
    return self.possible_actions

  def product(self, action):
    possible_space = self.production_space
    if action not in possible_space:
      raise WrongProductionException(f'"{action}" is not in action space.')
    self.node.production(action)

  def pretty_print(self, file=None):
    print('(', end='', file=file)
    self.start_node.pretty_print(file=file)
    print(')', file=file)

  def interprete(self, inputs):
    # pylint: disable=too-many-function-args
    return self.start_node.interprete([self.BITVECTOR(i) for i in inputs])

  @classmethod
  def parse(cls, program):
    return parse(program)

  def copy(self):
    return BitVectorLang(self.start_node.copy())

  @property
  def sequence(self):
    return self.start_node.sequence
  
  def is_complete(self):
    return self.start_node.is_complete()

  @classproperty
  @classmethod
  def TOKENS(cls):
    return sorted(ExprNode.TOKENS + BOPNode.TOKENS + ConstNode.TOKENS + ParamNode.TOKENS)

  @classmethod
  def tokens2prog(cls, tokens = []):
    pgm = cls()
    # pylint: disable=unsupported-membership-test
    for action in tokens:
      if action == 'neg':
        pgm.product(action)
      elif action == 'arith-neg':
        pgm.product(action)
      elif action in  ["+","-","x","/","%"] + ["||","&","^"]  + [">>_s",">>_u"]:
        pgm.product("bop")
        pgm.product(action)
      elif action in ConstNode.TOKENS:
        pgm.product("const")
        pgm.product(int(action))
      elif action in ParamNode.TOKENS:
        pgm.product("var")
        pgm.product(action)
    return pgm

  def is_const_pgm(self):
    return self.start_node.is_const_pgm()
      

######test######
if __name__ == '__main__':
  pgm = BitVectorLang.parse("( (  1 + 2 ) )")
  pgm.pretty_print()
