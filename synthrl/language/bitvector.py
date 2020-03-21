from synthrl.language.dsl import Tree
from synthrl.value import BitVector32

#N_z -> Var_z            => var
#     | Const_z          => const
#     | N_z Bop N_z      => bop
#     | -N_z             => neg
#     | Ite N_B N_z N_Z  => ite

class ExprNode(Tree):
  expr_productions = ['VAR_Z', 'CONST_Z', 'BOP', 'NEG', 'ITE']
  def production_space(self):
    if self.data =='hole':
      return self, self.expr_productions
    else:
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

#N_B -> true|false
#         | Nz=Nz | N_B land N_B |N_B lor N_B| N_B lnot N_B
class BOOLNode(Tree):
  bool_operations = ["true", "false", "equal","land","lor","lnot"]
  
# Bop -> + | − | & | ∥ | × | / |<< | >> | mod
class BOPNode(Tree):
  binary_operations = ["+", "-","&", "||", "x", "/", "<<", ">>","mod"]

  def production_space(self):
    if self.data=='hole':
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
    
#Const_z -> ...
class ConstNode(Tree):
  pass

#Var_z -> param1 | param2 ...
class ParamNode(Tree):
  param_space = ["param{}".format(i) for i in range(2)]

  def production_space(self):
    if self.data == 'hole':
      return self, self.param_space
    else:
      return self,[]
  
  def production(self, rule=None):
    if rule in self.param_space:
      self.data=rule
  
  def interprete(self, inputs): ##inputs as list?
    if self.data=="param1":
      return inputs[1]
    elif self.data=="param2":
      return inputs[2]
  
  def pretty_print(self):
    print('({})'.format(self.data), end='')
