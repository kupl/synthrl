#    P -> Expr
# Expr -> Var
#       | Cnst
#       | Bop
#       | - Expr
#       | ¬ Expr
#       | if Bool then Expr else Expr
#  Bop -> Expr + Expr
#       | Expr - Expr
#       | Expr * Expr
#       | Expr / Expr
#       | Expr % Expr
#       | Expr | Expr
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
#  Var -> param1
#       | param2

from synthrl.common.language.abstract.exception import SyntaxError
from synthrl.common.language.abstract.exception import WrongProductionException
from synthrl.common.language.abstract.lang import HOLE
from synthrl.common.language.abstract.lang import Program
from synthrl.common.language.abstract.lang import Tree
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
    if not cls.__BitVector or cls.__BitVector.size != cls.VECTOR_SIZE:
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
    print('(', end='', file=file)
    self.start_node.pretty_print(file=file)
    print(')', file=file)

  def interprete(self, inputs):
    # pylint: disable=too-many-function-args
    return self.start_node.interprete([self.BITVECTOR(i) for i in inputs])

  @classmethod
  def parse(cls, program):
    # remove outermost parentheses
    program = program.strip()[1:-1].strip()
    return cls(ExprNode.parse(program))

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
    return ExprNode.tokens + BOPNode.tokens + ConstNode.tokens + ParamNode.tokens


class ExprNode(Tree):
  # expr_productions = ['VAR_Z', 'CONST_Z', 'BOP', 'NEG', 'ITE']
  # expr_productions = ['var','const','bop','neg']
  expr_productions = ['bop','const','var', 'neg','arith-neg']
  # expr_productions += ['ite']
  def production_space(self):
    if self.data =='HOLE': 
      return self, self.expr_productions
    else:
      children=[]
      if self.data=="var":
        children=["VAR_Z"]
      elif self.data=="const":
        children=["CONST_Z"]
      elif self.data=="bop":
        children=["BOP"]
      elif self.data=="neg":
        children=["NEG"]
      elif self.data=="ite":
        children=['IF_BOOL', 'THEN_EXPR','ELSE_EXPR']
      for key in children:
        node, space = self.children[key].production_space()
        if len(space) > 0:
          return node, space
      return self, []
      
  def production(self, rule=None):
    if rule =="var":
      self.data="var"
      self.children = {
          'VAR_Z' : ParamNode()
      }
    if rule=="const":
      self.data="const"
      self.children ={
          'CONST_Z' : ConstNode()
      }
    if rule=="bop":
      self.data="bop"
      self.children={
          'BOP' : BOPNode()
      }
    if rule=="neg":
      self.data="neg"
      self.children={
        'NEG' : ExprNode()
      }
    if rule =="arith-neg":
      self.data=="arith-neg"
      self.children={
        'ARITH-NEG' : ExprNode()
      }
    
    # if rule=="ite":
    #   self.data="ite"
    #   self.children={
    #     "IF_BOOL" : BOOLNode(),
    #     "THEN_EXPR" : ExprNode(),
    #     "ELSE_EXPR" : ExprNode()
    #   }

  def interprete(self, inputs):
    if self.data =="var":
      return self.children['VAR_Z'].interprete(inputs)
    if self.data =="const":
      return self.children['CONST_Z'].interprete(inputs)
    if self.data =="bop":
      return self.children['BOP'].interprete(inputs)
    if self.data=="arith-neg":
      sub=self.children['ARITH-NEG'].interprete(inputs)
      return -sub
    if self.data =="neg": #logical neg
      return self.children['NEG'].interprete(inputs).logical_neg()
    # if self.data =="ite":
    #   pass

  def pretty_print(self,file=None):
    if self.data=="HOLE" or self.data=='hole':
      print(" (HOLE) ", end='', file=file)

    elif self.data == 'var':
      self.children['VAR_Z'].pretty_print(file=file)

    elif self.data=='const':
      self.children['CONST_Z'].pretty_print(file=file)

    elif self.data=='bop':
      self.children['BOP'].pretty_print(file=file)

    elif self.data=='neg':
      print("¬ ( ",end='', file=file)
      self.children['NEG'].pretty_print(file=file)

      print(" ) ",end='', file=file)
      
    elif self.data == 'arith-neg':
      print(" - ( ",end='', file=file)
      self.children['ARITH-NEG'].pretty_print(file=file)
      print(" ) ",end='', file=file)
    # elif self.data=="ite":
    #   print(' ( ',end='')
    #   print(" IF ",end='')
    #   self.children["IF_BOOL"].pretty_print(file=file)
    #   print(" THEN ",end='') 
    #   self.children["THEN_EXPR"].pretty_print(file=file)
    #   print(" ELSE ",end='') 
    #   self.children["ELSE_EXPR"].pretty_print(file=file)
    #   print(' ) ',end='')

  @classmethod
  def parse(cls, exp):
    # check if expression is string
    if not isinstance(exp, str):
      raise ValueError("Given expression {} is not a string".format(exp))
    if exp in ['param0', 'param1']: # var
      return ParamNode.parse(exp)
    elif exp in [str(i) for i in range(16+1)]: # const
      return ConstNode.parse(exp)
    elif exp.startswith('¬'): # neg
      # set operator
      op = 'neg'
      # get leftmost '('
      s = exp.find('(')
      # get rightmost ')'
      e = exp.rfind(')')
      # no parentheses
      if (s==-1 and e==-1):
        op_i = exp.find('¬')
        subexp = exp[op_i+1:].strip()
      # valid parentheses
      elif (s!=-1 and e!=-1):
        subexp = exp[s+1:e].strip()
      # invalid parentheses
      else:
        raise SyntaxError("Expression {} has invalid syntax.".format(exp))
      # set children
      children = {
        'NEG': ExprNode.parse(subexp)
      }
    elif exp.startswith('-'): # arith-neg
      # set operator
      op = 'arith-neg'
      # get leftmost '('
      s = exp.find('(')
      # get rightmost ')'
      e = exp.rfind(')')
      # no parentheses
      if (s==-1 and e==-1):
        op_i = exp.find('-')
        subexp = exp[op_i+1:].strip()
      # valid parentheses
      elif (s!=-1 and e!=-1):
        subexp = exp[s+1:e].strip()
      # invalid parentheses
      else:
        raise SyntaxError("Expression {} has invalid syntax.".format(exp))
      children = {
        'ARITH-NEG': ExprNode.parse(subexp)
      }
    else: # bop or error
      # set operator
      op = 'bop'
      subexp = exp
      children = {
        'BOP': BOPNode.parse(subexp.strip())
      }
    
    # create expression node
    node = cls(data=op)
    for key in children.keys():
      children[key].parent = node
    node.children = children
    return node

  @property
  def sequence(self):
    if self.data=="HOLE" or self.data=='hole':
      return []
    else:
      tokenized = []
      if self.data=='var':
        tokenized = tokenized + self.children['VAR_Z'].sequence
      elif self.data=='const':
        tokenized = tokenized + self.children['CONST_Z'].sequence
      elif self.data=='bop':
        tokenized = tokenized + self.children['BOP'].sequence
      elif self.data=='neg':
        tokenized.append(self.data)
        tokenized = tokenized + self.children['NEG'].sequence
      elif self.data=='arith-neg':
        tokenized.append(self.data)
        tokenized = tokenized + self.children['ARITH-NEG'].sequence
      return tokenized

  @classproperty
  @classmethod
  def tokens(cls):
    return ["arith-neg","neg"]
  
  def is_complete(self):
    if self.data=="HOLE" or self.data=="hole":
      return False
    else:
      is_comp= True
      for key in list(self.children.keys()):
        is_comp = True and (self.children[key].is_complete())
      return is_comp

#N_B -> true|false
#         | Nz=Nz | N_B land N_B |N_B lor N_B| N_B lnot N_B
#Add Later: <=_u
class BOOLNode(Tree):
  bool_operations = ["true", "false", "equal","land","lor","lnot"]

  def production_space(self):
    if self.data=='HOLE':
      return self, self.bool_operations
    else:
      if self.data == "true" or self.data=="false":
        return self, []
      else:
        for child in self.children:
          child_node = self.children[child]
          node, space = child_node.production_space()
          if len(space) > 0:
            return node, space
        return self, []

  def production(self,rule=None):
    if rule == "true" or rule == "false":
      self.data==rule
    if rule == "equal":
      self.data==rule
      self.children={
        "LeftExpr"  : ExprNode() ,
        "RightExpr" : ExprNode() 
      }
    if rule =="land" or rule =="lor":
      self.data=rule
      self.children={
        "LeftBool"  : BOOLNode() ,
        "RightBool" : BOOLNode() 
      }
    if rule=="lnot":
      self.data=rule
      self.children={
        "BOOL" : BOOLNode()
      }

  def interprete(self, inputs):
    if self.data=="true" :
      return True
    if self.data=="false":
      return False 
    if self.data=="equal":
      left  =  self.children["LeftExpr"].interprete(inputs)
      right = self.children["RightExpr"].interprete(inputs)
      return left==right
    if self.data=="land" :
      left  =  self.children["LeftBool"].interprete(inputs)
      right = self.children["RightBool"].interprete(inputs)
      return left and right
    if self.data=="lor" :
      left  =  self.children["LeftBool"].interprete(inputs)
      right = self.children["RightBool"].interprete(inputs)
      return left or right
    if self.data=="lnot":
      return not self.children["BOOL"].interprete(inputs)

  def pretty_print(self,file=None):
    if self.data=="HOLE" or self.data=="hole":
      print(" (HOLE) ", end="", file=file)
    elif self.data == "true" or self.data == "false":
      print(" {} ".format(self.data),end='', file=file)
    elif self.data == "lnot":
      print(" ~ " ,end='', file=file)
      print(' ( ', end='', file=file)
      self.children["BOOL"].pretty_print(file)
      print(' ) ', end='', file=file)
    else:
      print(' ( ',end='', file=file)
      if self.data == "equal":
        self.children["LeftExpr"].pretty_print(file=file)
        print(' {} '.format(self.data), end='', file=file)
        self.children["RightExpr"].pretty_print(file=file)
      else:
        self.children["LeftBool"].pretty_print(file=file)
        print(' {} '.format(self.data), end='', file=file)
        self.children["RightBool"].pretty_print(file=file)
      print(' ) ',end='', file=file)
      
  @classmethod
  def parse(cls, bexp):
    if not isinstance(bexp, str):
      raise ValueError("Boolean expression {} is not a string".format(bexp))
    
    if bexp=='true' or bexp=='false':
      return cls(data=bexp)
    
    def splitBool(bexp):
      stack = 0
      cur = 0
      while True: 
        if bexp[cur] == '(': 
          stack += 1
        elif bexp[cur] == ')': 
          stack -= 1
        elif bexp[cur] == '=':
          if stack == 0: 
            return (bexp[:cur].strip(), bexp[cur], bexp[cur + 1:].strip())
        elif bexp[cur:cur+2] in ['lo']: 
          if stack == 0: 
            return (bexp[:cur].strip(), bexp[cur:cur + 3], bexp[cur + 3:].strip())
        elif bexp[cur:cur+2] in ['la', 'ln']: 
          if stack == 0: 
            return (bexp[:cur].strip(), bexp[cur:cur + 4], bexp[cur + 4:].strip())
        else:
          pass
        cur += 1
    
    leftExpr, op, rightExpr = splitBool(bexp)
    if not op in ['=', 'land', 'lor', 'lnot']:
      raise SyntaxError("Invalid boolean operator {} is given.".format(op))
    
    if op=='=':
      op = 'equal' # for compatibility
      children = {
        'LeftEXPR': ExprNode.parse(leftExpr),
        'RightEXPR': ExprNode.parse(rightExpr)
      }
    elif op in ['land', 'lor', 'lnot']:
      children = {
        'LeftEXPR': BOOLNode.parse(leftExpr),
        'RightEXPR': BOOLNode.parse(rightExpr)
      }
    node = cls(data=op)
    for key in children.keys():
      children[key].parent = node
    node.children = children
    return node


# Bop -> {bitwise-logical opts} | {airthmetic opts}
# bitwise logical operators = {|, &, \oplus(XOR), singed>>, unsigned >>}
# airthmetic operators = {+, -. x , /, %} 
class BOPNode(Tree):
  binary_operations = ["+","-","x","/","%"] + ["||","&","^"] + [">>_s",">>_u"]

  def production_space(self):
    if self.data=='HOLE' or self.data=='hole':
      return self, self.binary_operations
    else:
      for key in ['LeftEXPR','RightEXPR']:
        node, space = self.children[key].production_space()
        if len(space) > 0:
          return node, space
      return self, []

  def production(self, rule=None): #rules should be one of in binary_operations
    self.data=rule
    self.children={
      'LeftEXPR'  :  ExprNode(),
      'RightEXPR' :  ExprNode()
    }

  def interprete(self, inputs):
    ##Arithmetics
    if self.data=="+":
      left=self.children['LeftEXPR'].interprete(inputs)
      right=self.children['RightEXPR'].interprete(inputs)
      return left+right
    if self.data=="-":
      left=self.children['LeftEXPR'].interprete(inputs)
      right=self.children['RightEXPR'].interprete(inputs)
      return left-right
    if self.data=="x":
      left=self.children['LeftEXPR'].interprete(inputs)
      right=self.children['RightEXPR'].interprete(inputs)
      return left*right
    if self.data=="/":
      left=self.children['LeftEXPR'].interprete(inputs)
      right=self.children['RightEXPR'].interprete(inputs)
      return left/right
    if self.data=="%":
      left=self.children['LeftEXPR'].interprete(inputs)
      right=self.children['RightEXPR'].interprete(inputs)
      return left%right
    ##bitwise, logic operators
    if self.data=="||":
      left=self.children['LeftEXPR'].interprete(inputs)
      right=self.children['RightEXPR'].interprete(inputs)
      return left|right
    if self.data=="&":
      left=self.children['LeftEXPR'].interprete(inputs)
      right=self.children['RightEXPR'].interprete(inputs)
      return left&right
    if self.data=="^":
      left=self.children['LeftEXPR'].interprete(inputs)
      right=self.children['RightEXPR'].interprete(inputs)
      return left^right
    ##shifts
    if self.data==">>_s":
      left=self.children['LeftEXPR'].interprete(inputs)
      right=self.children['RightEXPR'].interprete(inputs)
      return left>>right
    if self.data==">>_u":
      left=self.children['LeftEXPR'].interprete(inputs)
      right=self.children['RightEXPR'].interprete(inputs)
      return left.uns_rshift(right)
  
  def pretty_print(self,file=None):
    if self.data=="HOLE" or self.data=="hole":
      print(' ( {} ) '.format(self.data), file=file)
    else:
      print(' ( ', end='', file=file) 
      self.children['LeftEXPR'].pretty_print(file=file)
      print(' {} '.format(self.data), end='', file=file)
      self.children['RightEXPR'].pretty_print(file=file)
      print(' ) ', end='', file=file) 
    
  @classmethod
  def parse(cls, exp):
    # remove outermost parentheses
    exp = exp[1:-1].strip()
    # check if bop expression is string
    if not isinstance(exp, str):
      raise ValueError("Binary expression {} is not a string".format(exp))
    
    def splitBop(exp):
      stack = 0
      cur = 0
      while cur < len(exp):
        if exp[cur] == '(':
          stack += 1
        elif exp[cur] == ')':
          stack -= 1
        elif exp[cur] in ['+', '-', 'x', '/', '%', '&', '^']:
          if stack == 0: 
            return (exp[:cur].strip(), exp[cur], exp[cur + 1:].strip())
        elif exp[cur] == '|': 
          if stack == 0: 
            return (exp[:cur].strip(), exp[cur:cur + 2], exp[cur + 2:].strip())
        elif exp[cur] == '>': 
          if stack == 0: 
            return (exp[:cur].strip(), exp[cur:cur + 4], exp[cur + 4:].strip())
        else:
          pass
        cur += 1
      return exp, None, None
    
    leftExp, bop, rightExp = splitBop(exp)
    children = {
      'LeftEXPR': ExprNode.parse(leftExp),
      'RightEXPR': ExprNode.parse(rightExp)
    }
    # create binary expression node
    node = cls(data=bop)
    for key in children.keys():
      children[key].parent = node
    node.children = children
    return node
  
  @property
  def sequence(self):
    if self.data=="HOLE" or self.data=='hole':
      return []
    else:
      tokenized = []
      tokenized.append(self.data)
      tokenized = tokenized + self.children['LeftEXPR'].sequence + self.children['RightEXPR'].sequence
      return tokenized

  @classproperty
  @classmethod
  def tokens(cls):
    return cls.binary_operations

  def is_complete(self):
    if self.data=="HOLE" or self.data=="hole":
      return False
    else: 
      return (self.children['LeftEXPR'].is_complete()) and (self.children['RightEXPR'].is_complete())

#Const_z -> ...
class ConstNode(Tree):
  '''
  Following the standard in https://github.com/wslee/euphony/blob/master/benchmarks/bitvec/test/99_1000.sl,
  we set constant from 1 to 16, as 64-bitvecor
  #x0000000000000000
  #x0000000000000001
  #x0000000000000002
  #x0000000000000003
  #x0000000000000004
  #x0000000000000005
  #x0000000000000006
  #x0000000000000007
  #x0000000000000008
  #x0000000000000009
  #x000000000000000A
  #x000000000000000B
  #x000000000000000C
  #x000000000000000D
  #x000000000000000E
  #x000000000000000F
  #x0000000000000010
  '''
  constants = [i for i in range(16+1)]
  str_constants = [str(i) for i in range(16+1)]


  def __init__(self, *args, **kwargs):
    super(ConstNode, self).__init__(*args, **kwargs)
    # pylint: disable=too-many-function-args
    self.value = None if self.data == HOLE else BitVectorLang.BITVECTOR(self.data)

  def production_space(self):
    if self.data == 'HOLE'or self.data=='hole':
      return self, self.constants
    else:
      return self, []

  def production(self,rule):
    self.data=rule
    # pylint: disable=too-many-function-args
    self.value = BitVectorLang.BITVECTOR(self.data)

  def interprete(self, inputs):
    if self.data=="HOLE" or self.data=='hole':
      raise ValueError("The Constant Value is invalid as {}".format(self.data))
    return self.value

  def pretty_print(self,file=None):
    if self.data=="HOLE" or self.data=="hole":
      print(' (HOLE) ', end ='', file=file)
    else:
      print(' {} '.format(self.data),end='', file=file)

  @classmethod
  def parse(cls, const):
    # check if token is string
    if not isinstance(const, str):
      raise ValueError("Constant value {} is not a string.".format(const))
    # check if const is in [0,16]
    if not const in str(cls.constants):
      raise ValueError("Constant Value must be between 1 and 16, but {} is given.".format(const))
    
    return cls(const)

  @property
  def sequence(self):
    if self.data=="HOLE" or self.data=="hole":
      return []
    else:
      return [str(self.data)]

  @classproperty
  @classmethod
  def tokens(cls):
    return cls.str_constants

  def is_complete(self):
    if self.data=="HOLE" or self.data=="hole":
      return False
    else: 
      return True

#Var_z -> param1 | param2 ...
class ParamNode(Tree):
  param_space = ["param{}".format(i) for i in range(BitVectorLang.N_INPUT)]

  def production_space(self):
    if self.data == 'HOLE':
      return self, self.param_space
    else:
      return self,[]
  
  def production(self, rule=None):
    self.data=rule
  
  def interprete(self, inputs): ##inputs as list?
    return inputs[int(self.data[-1])]

  def pretty_print(self,file=None):
    if self.data=="HOLE" or self.data=="hole":
      print(' (HOLE) ', end ='', file=file)
    else:
      print(' {} '.format(self.data), end='', file=file)
      
  @classmethod
  def parse(cls, token):
    # check if token is string
    if not isinstance(token, str):
      raise ValueError("Parameter token {} is not a string".format(token))
    
    # check if token is in ['param0', 'param1']
    if not token in cls.param_space:
      raise SyntaxError("Invalid parameter token {} is given.".format(token))
    
    return cls(token)
  
  @property
  def sequence(self):
    if self.data=="HOLE" or self.data=="hole":
      return []
    else:
      return [self.data]
  
  @classproperty
  @classmethod
  def tokens(cls):
    return cls.param_space

  def is_complete(self):
    if self.data=="HOLE" or self.data=="hole":
      return False
    else: 
      return True
  