import os
import sys
# Abstract class that all environment classes should inherit
# class Environment:
#   def __init__(self):
#     raise NotImplementedError
  
#   def trainstion(self, action=None): #To Us, Transtion is deterministic
#     raise NotImplementedError

#   def rest(self):
#     raise NotImplementedError
class ListEnv:
  insts_rules     =["assign","assign_high", "assign_first_input", "assign_second_input","end"]
  func_rules      =["head" ,"last" ,"take" ,"drop" ,"access" ,"minimum" ,"maximum" ,"reverse" ,"sort" ,"sum"]
  high_func_rules =["map","filter","count","zipwit","scanl1"] 
  var_rules       =["a" , "b" , "c" , "d" , "e" , "f" , "h" , "i" , "j" ,"k" ,"l" ,"m" , "n" , "p"]
  input_rules     =["x","y"]
  output_rules    =["o"]
  intint_rules    =[ "(+1)","(-1)" ,"(*2)" ,"(/2)","(*(-1))" ,"(**2)","(*3)","(/3)","(*4)","(/4)"]
  intbool_rules   =["(>0)" ,"(<0)"  ,"(%2==0)"  ,"(%2==1)"]
  intintint_rules =["(+)", "(-)"   ,"(*)"   ,"MIN"   ,"MAX"]

  def __init__(self):
    pass
  
  def is_valid_sate(): #This method will characterize State Space
    '''
    Each State is (pp,pp',E)
      - pp parital program of Synthesizer     // 
      - pp' partial program of Verifier      //
      - E: Colleciton of Test cases         // 
    Each State wiil be defiend as dict().
    '''
    pass  
  def is_valid_action():
    pass
  def transition():
    pass

class Tree:
  def __init__(self,data,children):
    self.data=data
    self.children=children


class ListProgram(Tree):
  def __init__(self,data,children=list()):
    super().__init__(data,children)

  #apply production rule#
  def production():
    pass
  def __repr__():
    pass


current=ListProgram("start",None)