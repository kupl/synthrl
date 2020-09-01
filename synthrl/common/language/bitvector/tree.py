import os
from lark import Lark
from lark import Transformer

from synthrl.common.language.abstract.lang import Tree
from synthrl.common.language.abstract.lang import HOLE
from synthrl.common.utils import classproperty
import synthrl.common.language.bitvector.lang as lang


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
      elif self.data=="arith-neg":
        children=['ARITH-NEG']
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
      self.data="arith-neg"
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
      val = self.children['NEG'].interprete(inputs)
      return ~val
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
      print("~ ( ",end='', file=file)
      self.children['NEG'].pretty_print(file=file)
      print(" ) ",end='', file=file)
    elif self.data == 'arith-neg':
      print("neg ( ",end='', file=file)
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

  @property
  def sequence(self):
    seq = []
    if self.data == HOLE:
      seq.append(self.data)
    elif self.data=='var':
      seq = seq + self.children['VAR_Z'].sequence
    elif self.data=='const':
      seq = seq + self.children['CONST_Z'].sequence
    elif self.data=='bop':
      seq = seq + self.children['BOP'].sequence
    elif self.data=='neg':
      seq.append(self.data)
      seq = seq + self.children['NEG'].sequence
    elif self.data=='arith-neg':
      seq.append(self.data)
      seq = seq + self.children['ARITH-NEG'].sequence
    return seq

  @classproperty
  @classmethod
  def TOKENS(cls):
    return ["arith-neg","neg"]
  
  def is_complete(self):
    if self.data=="HOLE" or self.data=="hole":
      return False
    else:
      is_comp= True
      for key in list(self.children.keys()):
        is_comp = True and (self.children[key].is_complete())
      return is_comp

  def is_const_pgm(self):
    for key in list(self.children.keys()):
        is_comp = True and (self.children[key].is_const_pgm())
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
      return left.unsigned_rshift(right)
  
  def pretty_print(self,file=None):
    if self.data=="HOLE" or self.data=="hole":
      print(' ( {} ) '.format(self.data), file=file)
    else:
      print(' ( ', end='', file=file) 
      self.children['LeftEXPR'].pretty_print(file=file)
      print(' {} '.format(self.data), end='', file=file)
      self.children['RightEXPR'].pretty_print(file=file)
      print(' ) ', end='', file=file) 
    
  @property
  def sequence(self):
    seq = [self.data]
    if seq[-1] != HOLE:
      seq = seq + self.children['LeftEXPR'].sequence
    if seq[-1] != HOLE:
      seq = seq + self.children['RightEXPR'].sequence
    return seq

  @classproperty
  @classmethod
  def TOKENS(cls):
    return cls.binary_operations

  def is_complete(self):
    if self.data=="HOLE" or self.data=="hole":
      return False
    else: 
      return (self.children['LeftEXPR'].is_complete()) and (self.children['RightEXPR'].is_complete())
  
  def is_const_pgm(self):
    return True and (self.children['LeftEXPR'].is_const_pgm()) and (self.children['RightEXPR'].is_const_pgm())

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
    self.value = None if self.data == HOLE else lang.BitVectorLang.BITVECTOR(self.data)

  def production_space(self):
    if self.data == 'HOLE'or self.data=='hole':
      return self, self.constants
    else:
      return self, []

  def production(self,rule):
    self.data=rule
    # pylint: disable=too-many-function-args
    self.value = lang.BitVectorLang.BITVECTOR(self.data)

  def interprete(self, inputs):
    if self.data=="HOLE" or self.data=='hole':
      raise ValueError("The Constant Value is invalid as {}".format(self.data))
    return self.value

  def pretty_print(self,file=None):
    if self.data=="HOLE" or self.data=="hole":
      print(' (HOLE) ', end ='', file=file)
    else:
      print(' {} '.format(self.data),end='', file=file)

  @property
  def sequence(self):
    return [str(self.data)]

  @classproperty
  @classmethod
  def TOKENS(cls):
    return cls.str_constants

  def is_complete(self):
    if self.data=="HOLE" or self.data=="hole":
      return False
    else: 
      return True

  def is_const_pgm(self):
    return True

#Var_z -> param1 | param2 ...
class ParamNode(Tree):
  param_space = ["param{}".format(i) for i in range(lang.BitVectorLang.N_INPUT)]

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
      
  @property
  def sequence(self):
    return [self.data]
  
  @classproperty
  @classmethod
  def TOKENS(cls):
    return cls.param_space

  def is_complete(self):
    if self.data=="HOLE" or self.data=="hole":
      return False
    else: 
      return True

  def is_const_pgm(self):
    return False

# Transformer for lark-tree to bitvectorlang
class BitVectorTransformer(Transformer):
  def program(self, pgm):
    return lang.BitVectorLang(start_node=pgm[0])
  def expr(self, exp):
    # parentheses handling
    if isinstance(exp[0], ExprNode):
      return exp[0]
    # 1. const
    if isinstance(exp[0], ConstNode):
      return ExprNode(data="const", children={"CONST_Z": exp[0]})
    # 2. param
    elif isinstance(exp[0], ParamNode):
      return ExprNode(data="var", children={"VAR_Z": exp[0]})
    # 3. bop
    elif isinstance(exp[0], BOPNode):
      return ExprNode(data="bop", children={"BOP": exp[0]})
    # 4. ite
    elif isinstance(exp[0], BOOLNode):
      cond, b_true, b_false = exp
      return ExprNode(data="ite", children={'IF_BOOL': cond, 'THEN_EXPR': b_true,'ELSE_EXPR': b_false})
    # 5. neg or arith-neg
    else:
      op, exp = exp
      op = 'arith-neg' if op=='-' else 'neg'
      return ExprNode(data=op, children={op.upper(): exp})
  def bop(self, op):
    # parentheses handling
    if isinstance(op[0], BOPNode):
      return op[0]
    leftExpr, op, rightExpr = op[0], op[1], op[2]
    return BOPNode(data=op, children={"LeftEXPR": leftExpr, "RightEXPR": rightExpr})
  def const(self, const):
    return ConstNode(data=int(const[0].value))
  def var(self, param):
    return ParamNode(data=param[0].value)
  def bexpr(self, bexp):
    # parentheses handling
    if isinstance(bexp[0], BOOLNode) and len(bexp)==1:
      return bexp[0]
    # atomic values
    if bexp[0] in ['true', 'false']:
      return BOOLNode(data=bexp[0])
    # logical not
    elif isinstance(bexp[1], BOOLNode):
      return BOOLNode(data='lnot', children={"BOOL": bexp[1]})      
    # equality
    elif isinstance(bexp[0], ExprNode):
      return BOOLNode(data="equal", children={"LeftExpr": bexp[0], "RightExpr": bexp[2]})
    # boolean operations
    else:
      left, op, right = bexp
      return BOOLNode(data='l'+op, children={"LeftBool": left, "RightBool": right})
    
grammar = "grammar.lark"
bitvector_parser = Lark.open(grammar, start='program', parser='lalr', transformer=BitVectorTransformer())
def parser(program):
  return bitvector_parser.parse(program)
