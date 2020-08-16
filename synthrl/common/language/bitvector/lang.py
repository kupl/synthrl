#    P -> Expr
# Expr -> Var
#       | Cnst
#       | Expr + Expr
#       | Expr - Expr
#       | Expr * Expr
#       | Expr / Expr
#       | Expr % Expr
#       | Expr | Expr
#       | Expr & Expr
#       | Expr ^ Expr
#       | Expr >>s Expr
#       | Expr >>u Expr
#       | Expr << Expr
#       | - Expr
#       | ~ Expr
#       | if Bool then Expr else Expr
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
#  Var -> param1
#       | param2

import importlib

from synthrl.common.language.abstract.exception import SyntaxError
from synthrl.common.language.abstract.exception import WrongProductionException
from synthrl.common.language.abstract.lang import HOLE
from synthrl.common.language.abstract.lang import Program
from synthrl.common.language.abstract.lang import Tree
from synthrl.common.utils import classproperty


class BitVectorLang(Program):

  VECTOR_SIZE = 16
  __BitVector = None

  @classproperty
  @classmethod
  def BitVector(cls):
    if not cls.__BitVector or cls.__BitVector.size != cls.VECTOR_SIZE:
      bitvector = importlib.import_module('synthrl.common.value.bitvector')
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
    if action not in self.possible_actions:
      raise WrongProductionException(f'"{action}" is not in action space.')
    self.node.production(action)

  def pretty_print(self, file=None):
    self.start_node.pretty_print(file=file)
    print(file=file)

  def interprete(self, inputs):
    # pylint: disable=too-many-function-args
    return self.start_node.interprete([self.BitVector(i) for i in inputs])

  @classmethod
  def parse(cls, program):
    program = program.strip()
    return cls(ExprNode.parse(program))

  def copy(self):
    return BitVectorLang(self.start_node.copy())

class ExprNode(Tree):

  EXPR_TOKENS = ['var', 'const', '+', '-', '*', '/', '%', '|', '&', '^', '>>s', '>>u', '<<', '~', 'arith-neg']
  TRAVERSE_ORDER = {
    'var': ['VAR'],
    'const': ['CONST'],
    '+': ['LEFT', 'RIGHT'],
    '-': ['LEFT', 'RIGHT'],
    '*': ['LEFT', 'RIGHT'],
    '/': ['LEFT', 'RIGHT'],
    '%': ['LEFT', 'RIGHT'],
    '|': ['LEFT', 'RIGHT'],
    '&': ['LEFT', 'RIGHT'],
    '^': ['LEFT', 'RIGHT'],
    '>>s': ['LEFT', 'RIGHT'],
    '>>u': ['LEFT', 'RIGHT'],
    '<<': ['LEFT', 'RIGHT'],
    '~': ['CHILD'],
    'arith-neg': ['CHILD'],
    'ite': ['COND', 'THEN', 'ELSE']
  }
  BOP = {
    '+': lambda left, right: left + right,
    '-': lambda left, right: left - right,
    '*': lambda left, right: left * right,
    '/': lambda left, right: left / right,
    '%': lambda left, right: left % right,
    '|': lambda left, right: left | right,
    '&': lambda left, right: left & right,
    '^': lambda left, right: left ^ right,
    '>>s': lambda left, right: left >> right,
    '>>u': lambda left, right: left.unsigned_rshift(right),
    '<<': lambda left, right: left << right
  }
  UOP = {
    '~': lambda child: ~child,
    'arith-neg': lambda child: -child
  }

  def production_space(self):
    if self.data == HOLE:
      return self, self.EXPR_TOKENS
    else:
      for key in self.TRAVERSE_ORDER[self.data]:
        node, space = self.children[key].production_space()
        if len(space) > 0:
          return node, space
      else:
        return self, []

  def production(self, action):
    self.data = action
    if action == 'var':
      self.children.update(VAR=VarNode())
    elif action == 'const':
      self.children.update(CONST=CnstNode())
    elif action in ['+', '-', '*', '/', '%', '|', '&', '^', '>>s', '>>u', '<<']:
      self.children.update(LEFT=ExprNode(), RIGHT=ExprNode())
    elif action in ['~', 'arith-neg']:
      self.children.update(CHILD=ExprNode())
    else: # action == 'ite'
      self.children.update(COND=BoolNode(), THEN=ExprNode(), ELSE=ExprNode())

  def interprete(self, inputs):
    if self.data == 'var':
      return self.children['VAR'].interprete(inputs)
    elif self.data == 'const':
      return self.children['CONST'].interprete()
    elif self.data in ['+', '-', '*', '/', '%', '|', '&', '^', '>>s', '>>u', '<<']:
      left = self.children['LEFT'].interprete(inputs)
      right = self.children['RIGHT'].interprete(inputs)
      return self.BOP[self.data](left, right)
    elif self.data in ['~', 'arith-neg']:
      child = self.children['CHILD'].interprete(inputs)
      return self.UOP[self.data](child)
    else: # self.data == 'ite'
      cond = self.children['COND'].interprete(inputs)
      return self.children['THEN'].interprete(inputs) if cond else self.children['ELSE'].interprete(inputs)

  def pretty_print(self, file=None):
    if self.data == HOLE:
      print(f' ({self.data}) ', end='', file=file)
    elif self.data == 'var':
      self.children['VAR'].pretty_print(file=file)
    elif self.data == 'const':
      self.children['CONST'].pretty_print(file=file)
    elif self.data in ['+', '-', '*', '/', '%', '|', '&', '^', '>>s', '>>u', '<<']:
      print('( ', end='', file=file)
      self.children['LEFT'].pretty_print(file=file)
      print(f' {self.data} ', end='', file=file)
      self.children['RIGHT'].pretty_print(file=file)
      print(' ) ', end='', file=file)
    elif self.data == '~':
      print('~ (', end='', file=file)
      self.children['CHILD'].pretty_print(file=file)
      print(')', end='', file=file)
    elif self.data == 'arith-neg':
      print('- (', end='', file=file)
      self.children['CHILD'].pretty_print(file=file)
      print(')', end='', file=file)
    else: # self.data == 'ite'
      print('if ( ', end='', file=file)
      self.children['COND'].pretty_print(file=file)
      print(' ) then ( ', end='', file=file)
      self.children['THEN'].pretty_print(file=file)
      print(' ) else ( ', end='', file=file)
      self.children['ELSE'].pretty_print(file=file)
      print(' )', end='', file=file)

  @classmethod
  def parse(cls, expr):
    # Strip out whitespace and parentheses.
    expr = expr.strip()
    while expr[0] == '(' and expr[-1] == ')':
      expr = expr[1:-1].strip()

    # Var node.
    if expr in VarNode.VAR_TOKENS:
      return ExprNode('var', {'VAR': VarNode.parse(expr)})

    # Const node.
    elif expr.startswith('0x'):
      return ExprNode('const', {'CONST': CnstNode.parse(expr)})

    # ite.
    elif expr.startswith('if'):
      # Utility function.
      def split_if_then_else(expr):
        cur = 2 # Skip 'if'.
        stack = 0
        cond = None
        while cur < len(expr):
          if expr(cur) == '(':
            stack += 1
          elif expr[cur] == ')':
            stack -= 1
            if stack < 0:
              raise SyntaxError('Invalid syntax.')
          elif stack == 0:
            if not cond:
              if expr[cur:cur + 4] == 'then':
                cond = cur
            else: # not then
              if expr[cur:cur + 4] == 'else':
                return expr[:cond], expr[cond + 4:cur], expr[cur + 4:] 
          cur += 1

      # Split condition, then, and else.
      cond, then, els = split_if_then_else(expr)
      return ExprNode('ite', {'COND': BoolNode.parse(cond), 'THEN': ExprNode.parse(then), 'ELSE': ExprNode.parse(els)})

    # UOPs.
    elif expr[0] in ['-', '~']:
      op = expr[0]
      return ExprNode(op if op == '~' else 'arith-neg', {'CHILD': ExprNode.parse(expr[1:])})

    # BOPs.
    else:
      # Utility function.
      def split_left_op_right(expr):
        stack = 0
        cur = 0
        while cur < len(expr):
          if expr[cur] == '(':
            stack += 1
          elif expr[cur] == ')':
            stack -= 1
            if stack < 0:
              raise SyntaxError('Invalid syntax.')
          elif stack == 0:
            if expr[cur] in ['+', '-', '*', '/', '%', '|', '&', '^']:
              return expr[:cur], expr[cur], expr[cur + 1:]
            elif expr[cur:cur + 3] in ['>>s', '>>u']:
              return expr[:cur], expr[cur:cur + 3], expr[cur + 3:]
            elif expr[cur:cur + 2] in ['<<']:
              return expr[:cur], expr[cur:cur + 2], expr[cur + 2:]
          cur += 1
        raise SyntaxError('Invalid syntax.')
      
      # Split expression into left, operator, and right.
      left, op, right = split_left_op_right(expr)

      # Create and return a node
      return ExprNode(op, {'LEFT': ExprNode.parse(left), 'RIGHT': ExprNode.parse(right)})
    

class BoolNode(Tree):

  BOOL_TOKENS = ['true', 'false', '=', 'and', 'or', 'not']
  TRAVERSE_ORDER = {
    'true': [],
    'false': [],
    '=': ['LEFT', 'RIGHT'],
    'and': ['LEFT', 'RIGHT'],
    'or': ['LEFT', 'RIGHT'],
    'not': ['CHILD']
  }
  BOOL = {
    'true': True,
    'false': False
  }
  BOP = {
    '=': lambda left, right: left == right,
    'and': lambda left, right: left and right,
    'or': lambda left, right: left or right,
  }
  UOP = {
    'not': lambda child: not child
  }

  def production_space(self):
    if self.data == HOLE:
      return self, self.BOOL_TOKENS
    else:
      for key in self.TRAVERSE_ORDER[self.data]:
        node, space = self.children[key].production_space()
        if len(space) > 0:
          return node, space
      else:
        return self, []

  def production(self, action):
    self.data = action
    if action == '=':
      self.children.update(LEFT=ExprNode(), RIGHT=ExprNode())
    elif action in ['and', 'or']:
      self.children.update(LEFT=BoolNode(), RIGHT=BoolNode())
    else: # action == 'not'
      self.children.update(CHILD=BoolNode())

  def interprete(self, inputs):
    if self.data in ['true', 'false']:
      return self.BOOL[self.data]
    elif self.data in ['=', 'and', 'or']:
      left = self.data['LEFT'].interprete(inputs)
      right = self.data['RIGHT'].interprete(inputs)
      return self.BOP[self.data](left, right)
    else:
      child = self.data['CHILD'].interprete(inputs)
      return self.UOP[self.data](child)

  def pretty_print(self, file=None):
    if self.data == HOLE:
      print(' (HOLE) ', end='', file='')
    elif self.data in ['true', 'false']:
      print(f' {self.data} ', end='', file='')
    elif self.data in ['=', 'and', 'or']:
      print('( ', end='', file='')
      self.children['LEFT'].pretty_print(file=file)
      print(f') {self.data} (', end='', file=file)
      self.children['RIGHT'].pretty_print(file=file)
      print(')', end='', file=file)
    else: # self.data == 'not'
      print('not (', end='', file=file)
      self.children['CHILD'].pretty_print(file=file)
      print(')', end='', file=file)

  @classmethod
  def parse(cls, bexp):
    # Strip out whitespace and parentheses.
    bexp = bexp.strip()
    while bexp[0] == '(' and bexp[-1] == ')':
      bexp = bexp[1:-1].strip()

    # Simple true and false.
    if bexp in ['true', 'false']:
      return BoolNode(bexp)

    # UOP not.
    elif bexp.startswith('not'):
      return BoolNode('not', {'CHILD': BoolNode.parse(bexp[3:])})

    # Other BOPs.
    else:
      # Utility function.
      def split_left_op_right(bexp):
        stack = 0
        cur = 0
        while cur < len(bexp):
          if bexp[cur] == '(':
            stack += 1
          elif bexp[cur] == ')':
            stack -= 1
            if stack < 0:
              raise SyntaxError('Invalid syntax.')
          elif stack == 0:
            if bexp[cur] == '=':
              return bexp[:cur], bexp[cur], bexp[cur + 1:]
            elif bexp[cur:cur + 3] == 'and':
              return bexp[:cur], bexp[cur:cur + 3], bexp[cur + 3:]
            elif bexp[cur:cur + 2] == 'or':
              return bexp[:cur], bexp[cur:cur + 2], bexp[cur + 2:]
          cur += 1
        raise SyntaxError('Invalid syntax.')

      # Split expression into left, operator, and right.
      left, op, right = split_left_op_right(bexp)
      
      # Create and return a node.
      if op == '=':
        children = {
          'LEFT': ExprNode.parse(left),
          'RIGHT': ExprNode.parse(right)
        }
      else: # op in ['and', 'or']
        children = {
          'LEFT': BoolNode.parse(left),
          'RIGHT': BoolNode.parse(right)
        }
      return BoolNode(op, children)

class VarNode(Tree):

  VAR_TOKENS = ['param0', 'param1']

  def production_space(self):
    return self, (self.VAR_TOKENS if self.data == HOLE else [])

  def production(self, action):
    self.data = action

  def interprete(self, inputs):
    return inputs[int(self.data[-1])]

  def pretty_print(self, file=None):
    print(' (HOLE) ' if self.data == HOLE else f' {self.data} ', end='', file=file)

  @classmethod
  def parse(cls, var):
    return VarNode(var)

class CnstNode(Tree):

  CNST_TOKENS = ['0x00', '0x01', '0x02', '0x03', '0x04', '0x05', '0x06', '0x07', '0x08', '0x09', '0x0A', '0x0B', '0x0C', '0x0D', '0x0E', '0x0F', '0x10']

  def __init__(self, *args, **kwargs):
    super(CnstNode, self).__init__(*args, **kwargs)
    # pylint: disable=too-many-function-args
    self.value = None if self.data == HOLE else BitVectorLang.BitVector(self.data)

  def production_space(self):
    return self, (self.CNST_TOKENS if self.data == HOLE else [])

  def production(self, action):
    self.data = action
    # pylint: disable=too-many-function-args
    self.value = BitVectorLang.BitVector(self.data)

  def interprete(self):
    return self.value

  def pretty_print(self, file=None):
    print(' (HOLE) ' if self.data == HOLE else f' {self.data} ', end='', file=file)

  @classmethod
  def parse(cls, const):
    return CnstNode(const)
