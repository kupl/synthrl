from synthrl.language.abstract import Node
from synthrl.language.abstract import Tree
from synthrl.value.bitvector import BitVector64
from synthrl.language.abstract import WrongProductionException

#N_z -> Var_z            => var
#     | Const_z          => const
#     | N_z Bop N_z      => bop
#     | -N_z             => neg
#     | Ite N_B N_z N_Z  => ite

#'bop', 'neg', 'ite' as instruction command
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
#N_z
class ExprNode(Node):
  # expr_productions = ['VAR_Z', 'CONST_Z', 'BOP', 'NEG', 'ITE']
  # expr_productions = ['var','const','bop','neg']
  expr_productions = ['bop','const','var']
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
    if rule=="ite":
      self.data="ite"
      self.children={
        "IF_BOOL" : BOOLNode(parent=self),
        "THEN_EXPR" : ExprNode(parent=self),
        "ELSE_EXPR" : ExprNode(parent=self)
      }

  def interprete(self, inputs):
    if self.data =="var":
      return self.children['VAR_Z'].interprete(inputs)
    if self.data =="const":
      pass
    if self.data =="bop":
      pass
    if self.data =="neg":
      sub=self.children['NEG'].interprete(inputs)
      return sub.negate()
    if self.data =="ite":
      pass

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
    elif self.data=="ite":
      print(' ( ',end='')
      print(" IF ",end='')
      self.children["IF_BOOL"].pretty_print(file=file)
      print(" THEN ",end='') 
      self.children["THEN_EXPR"].pretty_print(file=file)
      print(" ELSE ",end='') 
      self.children["ELSE_EXPR"].pretty_print(file=file)
      print(' ) ',end='')

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


# Bop -> + | − | & | ∥ | × | / |<< | >> | mod
#Add Later : XOR
class BOPNode(Node):
  binary_operations = ["+", "-","&", "||", "x", "/", "<<", ">>","mod"]

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
    if self.data=="+":
      left=self.children['LeftEXPR'].interprete(inputs)
      right=self.children['RightEXPR'].interprete(inputs)
      return left+right
    if self.data=="-":
      left=self.children['LeftEXPR'].interprete(inputs)
      right=self.children['RightEXPR'].interprete(inputs)
      return left-right
    if self.data=="&":
      left=self.children['LeftEXPR'].interprete(inputs)
      right=self.children['RightEXPR'].interprete(inputs)
      return left&right
    if self.data=="||":
      left=self.children['LeftEXPR'].interprete(inputs)
      right=self.children['RightEXPR'].interprete(inputs)
      return left|right
    if self.data=="x":
      left=self.children['LeftEXPR'].interprete(inputs)
      right=self.children['RightEXPR'].interprete(inputs)
      return left*right
    if self.data=="/":
      pass
    if self.data=="<<":
      pass
    if self.data==">>":
      pass
    if self.data=="mod":
      pass
  
  def pretty_print(self,file=None):
    print(' ( ', end='') 
    self.children['LeftEXPR'].pretty_print(file=file)
    print(' {} '.format(self.data), end='')
    self.children['RightEXPR'].pretty_print(file=file)
    print(' ) ', end='') 


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
  def production_space(self):
    if self.data == 'HOLE'or self.data=='hole':
      return self, self.constants
    else:
      return self, []

  def production(self,rule):
    self.data=rule

  def interprete(self, inputs):
    return BitVector64(self.data)

  def pretty_print(self,file=None):
    if self.data=="HOLE" or self.data=="hole":
      print(' (HOLE) ', end ='')
    else:
      print(' {} '.format(self.data),end='')

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
    if self.data=="param1":
      return BitVector64(inputs[1])
    elif self.data=="param2":
      return BitVector64(inputs[2])
  
  def pretty_print(self,file=None):
    if self.data=="HOLE" or self.data=="hole":
      print(' (HOLE) ', end ='')
    else:
      print(' {} '.format(self.data), end='')
  
######test######
# if __name__ == '__main__':
#   print("--test--")
#   vec_program = BitVectorLang()
#   poss = vec_program.production_space()
#   print(poss)
#   vec_program.production('bop')
#   poss = vec_program.production_space()
#   print(poss)
#   vec_program.production('+')
#   vec_program.pretty_print()
#   print(vec_program.production_space())
#   vec_program.production('const')
#   vec_program.pretty_print()
#   print(vec_program.production_space())
#   vec_program.production(10)
#   vec_program.pretty_print()
#   print(vec_program.production_space())
#   vec_program.production('neg')
#   vec_program.pretty_print()
#   print(vec_program.production_space())
