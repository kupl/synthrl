from synthrl.language.abstract import Node
from synthrl.language.abstract import Tree
from synthrl.value.bitvector import BitVector64
from synthrl.value.bitvector import BitVector32
from synthrl.value.bitvector import BitVector16
from synthrl.language.abstract import WrongProductionException
from synthrl.utils.decoratorutils import classproperty

#N_z -> Var_z            => var
#     | Const_z          => const
#     | N_z Bop N_z      => bop
#     | - N_z            => arithmethic neg
#     | ¬ N_z            => bitwise_neg   
# out-of-range, for now
#     | Ite N_B N_z N_Z  => ite

#[16,32,64]
VECTOR_LENGTH = 32 

class BitVectorLang(Tree):
  def __init__(self):
    self.start_node = ExprNode()

  def production_space(self):
    _ , space = self.start_node.production_space()
    return space

  def production(self,action):
    node, possible_actions = self.start_node.production_space()
    if action in possible_actions:
        node.production(action)
    else:
      raise WrongProductionException('Invalid production rule "{}" is given.'.format(action))
      
  def pretty_print(self,file=None):
    print('(',end='')
    self.start_node.pretty_print(file=file)
    print(')')

  def interprete(self, inputs):
    return self.start_node.interprete(inputs)
  
  @classmethod
  def parse(cls, program):
    # check if program is string
    if not isinstance(program, str):
      raise ValueError("Program {} is not a string.".format(program))
    
    # remove outermost parentheses
    program = program.strip()[1:-1].strip()
    return ExprNode.parse(program)

  def tokenize(self):
    return self.start_node.tokenize()
  
  def is_complete(self):
    return self.start_node.is_complete()

  @classproperty
  @classmethod
  def tokens(cls):
    return ExprNode.tokens + BOPNode.tokens + ConstNode.tokens + ParamNode.tokens
#N_z
class ExprNode(Node):
  # expr_productions = ['VAR_Z', 'CONST_Z', 'BOP', 'NEG', 'ITE']
  # expr_productions = ['var','const','bop','neg']
  expr_productions = ['bop','const','var', 'neg']
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
          'VAR_Z' : ParamNode(parent=self)
      }
    if rule=="const":
      self.data="const"
      self.children ={
          'CONST_Z' : ConstNode(parent=self)
      }
    if rule=="bop":
      self.data="bop"
      self.children={
          'BOP' : BOPNode(parent=self)
      }
    if rule=="neg":
      self.data="neg"
      self.children={
        'NEG' : ExprNode(parent=self)
      }
    if rule =="arith-neg":
      self.data=="arith-neg"
      self.children={
        'ARITH-NEG' : ExprNode(parent=self)
      }
    
    # if rule=="ite":
    #   self.data="ite"
    #   self.children={
    #     "IF_BOOL" : BOOLNode(parent=self),
    #     "THEN_EXPR" : ExprNode(parent=self),
    #     "ELSE_EXPR" : ExprNode(parent=self)
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
      print(" (HOLE) ", end='')
    elif self.data == 'var':
      self.children['VAR_Z'].pretty_print(file=file)
    elif self.data=='const':
      self.children['CONST_Z'].pretty_print(file=file)
    elif self.data=='bop':
      self.children['BOP'].pretty_print(file=file)
    elif self.data=='neg':
      print("NEG ( ",end='')
      self.children['NEG'].pretty_print(file=file)
      print(" ) ",end='')
    elif self.data == 'arith-neg':
      print("ARITH-NEG ( ",end='')
      self.children['ARITH-NEG'].pretty_print(file=file)
      print(" ) ",end='')
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

  def tokenize(self):
    if self.data=="HOLE" or self.data=='hole':
      return []
    else:
      tokenized = []
      if self.data=='var':
        tokenized = tokenized + self.children['VAR_Z'].tokenize()
      elif self.data=='const':
        tokenized = tokenized + self.children['CONST_Z'].tokenize()
      elif self.data=='bop':
        tokenized = tokenized + self.children['BOP'].tokenize()
      elif self.data=='neg':
        tokenized.append(self.data)
        tokenized = tokenized + self.children['NEG'].tokenize()
      elif self.data=='arith-neg':
        tokenized.append(self.data)
        tokenized = tokenized + self.children['ARITH-NEG'].tokenize()
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
class BOOLNode(Node):
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
        "LeftExpr"  : ExprNode(parent=self) ,
        "RightExpr" : ExprNode(parent=self) 
      }
    if rule =="land" or rule =="lor":
      self.data=rule
      self.children={
        "LeftBool"  : BOOLNode(parent=self) ,
        "RightBool" : BOOLNode(parent=self) 
      }
    if rule=="lnot":
      self.data=rule
      self.children={
        "BOOL" : BOOLNode(parent=self)
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
      print(" (HOLE) ", end="")
    elif self.data == "true" or self.data == "false":
      print(" {} ".format(self.data),end='')
    elif self.data == "lnot":
      print(" ~ " ,end='')
      print(' ( ', end='')
      self.children["BOOL"].pretty_print(file)
      print(' ) ', end='')
    else:
      print(' ( ',end='')
      if self.data == "equal":
        self.children["LeftExpr"].pretty_print(file=file)
        print(' {} '.format(self.data), end='')
        self.children["RightExpr"].pretty_print(file=file)
      else:
        self.children["LeftBool"].pretty_print(file=file)
        print(' {} '.format(self.data), end='')
        self.children["RightBool"].pretty_print(file=file)
      print(' ) ',end='')
      
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
class BOPNode(Node):
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
      'LeftEXPR'  :  ExprNode(parent=self),
      'RightEXPR' :  ExprNode(parent=self)
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
      print(' ( {} ) '.format(self.data))
    else:
      print(' ( ', end='') 
      self.children['LeftEXPR'].pretty_print(file=file)
      print(' {} '.format(self.data), end='')
      self.children['RightEXPR'].pretty_print(file=file)
      print(' ) ', end='') 
    
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
  
  def tokenize(self):
    if self.data=="HOLE" or self.data=='hole':
      return []
    else:
      tokenized = []
      tokenized.append(self.data)
      tokenized = tokenized + self.children['LeftEXPR'].tokenize() + self.children['RightEXPR'].tokenize()
      return tokenized

  @classproperty
  @classmethod
  def tokens(cls):
    return cls.binary_operations

  def is_complete(self):
    if self.data=="HOLE" or self.data=="hole":
      return False
    else: 
      return True and (self.children['LeftEXPR'].is_complete()) and (self.children['RightEXPR'].is_complete())

#Const_z -> ...
class ConstNode(Node):
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
  def production_space(self):
    if self.data == 'HOLE'or self.data=='hole':
      return self, self.constants
    else:
      return self, []

  def production(self,rule):
    self.data=rule

  def interprete(self, inputs):
    if self.data=="HOLE" or self.data=='hole':
      raise ValueError("The Constant Value is invalid as {}".format(self.data))

    if VECTOR_LENGTH==16:    
      return BitVector16(self.data)
    elif VECTOR_LENGTH ==32 :
      return BitVector32(self.data)

  def pretty_print(self,file=None):
    if self.data=="HOLE" or self.data=="hole":
      print(' (HOLE) ', end ='')
    else:
      print(' {} '.format(self.data),end='')

  @classmethod
  def parse(cls, const):
    # check if token is string
    if not isinstance(const, str):
      raise ValueError("Constant value {} is not a string.".format(const))
    # check if const is in [0,16]
    if not const in str(cls.constants):
      raise ValueError("Constant Value must be between 1 and 16, but {} is given.".format(const))
    
    return cls(const)

  def tokenize(self):
    if self.data=="HOLE" or self.data=="hole":
      return []
    else:
      return [self.data]

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
class ParamNode(Node):
  param_space = ["param{}".format(i) for i in range(2)]

  def production_space(self):
    if self.data == 'HOLE':
      return self, self.param_space
    else:
      return self,[]
  
  def production(self, rule=None):
    self.data=rule
  
  def interprete(self, inputs): ##inputs as list?
    if self.data=="param0":
      if VECTOR_LENGTH==16:
        return BitVector16(inputs[0])
      elif VECTOR_LENGTH==32:
        return BitVector32(inputs[0])
    elif self.data=="param1":
      if VECTOR_LENGTH==16:
        return BitVector16(inputs[1])
      elif VECTOR_LENGTH==32:
        return BitVector32(inputs[1])

  def pretty_print(self,file=None):
    if self.data=="HOLE" or self.data=="hole":
      print(' (HOLE) ', end ='')
    else:
      print(' {} '.format(self.data), end='')
      
  @classmethod
  def parse(cls, token):
    # check if token is string
    if not isinstance(token, str):
      raise ValueError("Parameter token {} is not a string".format(token))
    
    # check if token is in ['param0', 'param1']
    if not token in cls.param_space:
      raise SyntaxError("Invalid parameter token {} is given.".format(token))
    
    return cls(token)
  
  def tokenize(self):
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

######test######
#if __name__ == '__main__':
  #print("--test--")
  #vec_program = BitVectorLang()
  #print(vec_program.production_space())
  #vec_program.production('bop')
  #print(vec_program.production_space())
  #vec_program.production('+')
  #vec_program.pretty_print()
  #print(vec_program.production_space())
  #vec_program.production('const')
  #vec_program.pretty_print()
  #print(vec_program.production_space())
  #vec_program.production(10)
  #vec_program.pretty_print()
  #print(vec_program.production_space())
  #vec_program.production('neg')
  #vec_program.pretty_print()
  #print(vec_program.production_space())
  #vec_program.production('var')
  #vec_program.pretty_print()
  #print(vec_program.production_space())
