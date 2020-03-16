import os
import sys

class ListEnv:   
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
  def __init__(self,data,children=list()):
    self.data=data
    self.children=children

  def append_children(self,new_children):
    self.children.extend(new_children)

  def __repr__(self):
    return 'Tree(%s, %s)' % (self.data, self.children)

  def traverse(self, target_data):
    for child in self.children:
      temp=child.traverse(target_data) 
      if child.traverse(target_data) != None:
        return temp
    if self.data==target_data:
      return self
    # if self.data == target_data:
    #   return self
    # else:
    #   for child in self.children:
    #     temp=child.traverse(target_data) 
    #     if child.traverse(target_data) != None:
    #       return temp
         

    # else:
    #   for child in (self.children):
    #     if child.data==target_data:
    #       return child
        
  def is_child_exist(self,target_data):
    for child in self.children:
      if child.data==target_data:
        return child
    return None    


class ListProgram(Tree):
  insts_rules     =["assign","assign_high", "assign_first_input", "assign_second_input","end"]
  func_rules      =["head" ,"last" ,"take" ,"drop" ,"access" ,"minimum" ,"maximum" ,"reverse" ,"sort" ,"sum"]
  high_func_rules =["map","filter","count","zipwith","scanl1"] 
  var_rules       =["a" , "b" , "c" , "d" , "e" , "f" , "h" , "i" , "j" ,"k" ,"l" ,"m" , "n" , "p"] + ["o"]
  #For moments a variable being used for arugment 
  arugments_rules =["a" , "b" , "c" , "d" , "e" , "f" , "h" , "i" , "j" ,"k" ,"l" ,"m" , "n" , "p"] + ["x","y"]
  intint_rules    =[ "(+1)","(-1)" ,"(*2)" ,"(/2)","(*(-1))" ,"(**2)","(*3)","(/3)","(*4)","(/4)"]
  intbool_rules   =["(>0)" ,"(<0)"  ,"(%2==0)"  ,"(%2==1)"]
  intintint_rules =["(+)", "(-)"   ,"(*)"   ,"MIN"   ,"MAX"]

  #I will set data of each node in tree as Dictionary to involve:
  # which is
  #{
  #   Name of Production Rule (Variable) :  ____,
  #   The Token Generated by Upper Production Rule: ____,
  # }

  def __init__(self,data,children=list()):
    super().__init__(data,children)
  
  def production_inst(self,action):
    if action=="assign":
      self.append_children( [Tree({"ASSIGN":"assign"}   , [Tree({"VAR":"HOLE"}), Tree({"FUNC":"HOLE"})]   )])
    if action=="assign_high":
      self.append_children( [Tree({"ASSIGN":"assign_high"}   , [Tree({"VAR":"HOLE"}), Tree({"HIGH_FUNC":"HOLE"})]   )  ] )
    elif action=="assign_first_input" :
      self.append_children(    [Tree({"ASSIGN":"assign"}       ,        [Tree({"VAR":"HOLE"}), Tree({"VAR":"x"})]    )   ]    )
    elif action=="assign_first_input" :
      self.append_children( [Tree({"ASSIGN":"assign"}       ,        [Tree({"VAR":"HOLE"}), Tree({"VAR":"y"})]    )   ]  )

  def production_func(self,action):
    if action=="head":
      current_node = self.traverse({"FUNC" : "HOLE"})
      current_node.data={"FUNC" : "HEAD"}
      current_node.children=[Tree({"VAR":"HOLE"})]
    elif action=="last":
      current_node = self.traverse({"FUNC" : "HOLE"})
      current_node.data={"FUNC" : "LAST"}
      current_node.children=[Tree({"VAR":"HOLE"})]
    elif action=="take":
      current_node = self.traverse({"FUNC" : "HOLE"})
      current_node.data={"FUNC" : "TAKE"}
      current_node.children=[Tree({"VAR":"HOLE"}),Tree({"VAR":"HOLE"})]
    elif action=="drop":
      current_node = self.traverse({"FUNC" : "HOLE"})
      current_node.data={"FUNC" : "DROP"}
      current_node.children=[Tree({"VAR":"HOLE"}),Tree({"VAR":"HOLE"}) ]
    elif action=="access":
      current_node = self.traverse({"FUNC" : "HOLE"})
      current_node.data={"FUNC" : "ACCESS"}
      current_node.children=[Tree({"VAR":"HOLE"}),Tree({"VAR":"HOLE"}) ]
    elif action=="minimum":
      current_node = self.traverse({"FUNC" : "HOLE"})
      current_node.data={"FUNC" : "MINIMUM"}
      current_node.children=[Tree({"VAR":"HOLE"})]
    elif action=="maximum":
      current_node = self.traverse({"FUNC" : "HOLE"})
      current_node.data={"FUNC" : "MAXIMUM"}
      current_node.children=[Tree({"VAR":"HOLE"})]
    elif action=="reverse":
      current_node = self.traverse({"FUNC" : "HOLE"})
      current_node.data={"FUNC" : "REVERSE"}
      current_node.children=[Tree({"VAR":"HOLE"})]
    elif action=="sort":
      current_node = self.traverse({"FUNC" : "HOLE"})
      current_node.data={"FUNC" : "SORT"}
      current_node.children=[Tree({"VAR":"HOLE"})]
    elif action=="sum":
      current_node = self.traverse({"FUNC" : "HOLE"})
      current_node.data={"FUNC" : "SUM"}
      current_node.children=[Tree({"VAR":"HOLE"})]

  def production_func_high(self, action):
    if action=="map":
      current_node = self.traverse({"HIGH_FUNC" : "HOLE"})
      current_node.data={"HIGH_FUNC" : "MAP"}
      current_node.children=[ Tree({"INTINIT":"HOLE"})  ,  Tree({"VAR":"HOLE"})]
    elif action=="filter":
      current_node = self.traverse({"HIGH_FUNC" : "HOLE"})
      current_node.data={"HIGH_FUNC" : "FILTER"}
      current_node.children=[Tree({"INTBOOL":"HOLE"})  ,  Tree({"VAR":"HOLE"})]
    elif action=="count":
      current_node = self.traverse({"HIGH_FUNC" : "HOLE"})
      current_node.data={"HIGH_FUNC" : "COUNT"}
      current_node.children=[Tree({"INTBOOL":"HOLE"})  ,  Tree({"VAR":"HOLE"})]
    elif action=="zipwith":
      current_node = self.traverse({"HIGH_FUNC" : "HOLE"})
      current_node.data={"HIGH_FUNC" : "ZIPWITH"}
      current_node.children=[Tree({"INTINTINT":"HOLE"})  ,  Tree({"VAR":"HOLE"}),Tree({"VAR":"HOLE"})]
    elif action=="scanl1":
      current_node = self.traverse({"HIGH_FUNC" : "HOLE"})
      current_node.data={"HIGH_FUNC" : "SCANL1"}
      current_node.children=[Tree({"INTINTINT":"HOLE"})  ,  Tree({"VAR":"HOLE"}),Tree({"VAR":"HOLE"})]
  
  def production_INTINT(self,intint_func):
    current_node=self.traverse({"INTINIT":"HOLE"})
    current_node.data={"INTINIT":intint_func}
  def production_INTBOOL(self,intbool_func):
    current_node=self.traverse({"INTBOOL":"HOLE"})
    current_node.data={"INTBOOL":intbool_func}
  def production_INTINTINT(self,intintint_bool):
    current_node=self.traverse({"INTINTINT":"HOLE"})
    current_node.data={"INTINTINT":intintint_bool}
    

  def production_arg(self,arg_var):
    for x in self.func_rules:
      current_node = self.traverse({"FUNC" : x.upper()})
      if current_node != None:
        target_child =current_node.is_child_exist({"VAR":"HOLE"})
        if target_child != None:
           target_child.data={"VAR":arg_var}
  
  def production_var_assign(self,var):    
    for insts in self.children:
      if insts.children[0].data == {"VAR":"HOLE"}:
        insts.children[0].data = {"VAR":var}
        return

  def prog_to_string(self):
    if self.data=={"START": "start"}: pass
    prog_string=""
    for child in self.children:
      # print("******")
      # print(child.children[1])
      # print("******")
      if child.data=={"ASSIGN":"assign"}: 
        prog_string =  prog_string + child.children[0].data["VAR"]  + " <- " +  child.children[1].data["FUNC"] +  " "+ self.prog_to_string_aux(child.children[1])  + " ;"  + "\n" 
      elif child.data=={"ASSIGN":"assign_high"}:
        prog_string = prog_string + child.children[0].data["VAR"]  + "  <-  " +  child.children[1].data["HIGH_FUNC"] + " " +  self.prog_to_string_aux(child.children[1])  + " ;"  + "\n" 
    return prog_string
  
  def prog_to_string_aux(self,tree):
    return_str = ""
    for child in tree.children:
      if "VAR" in child.data:
        return_str = return_str + child.data["VAR"] + " "
      elif "INTINT" in child.data:
        return_str = return_str+child.data["INTINT"] + " "
      elif "INTBOOL" in child.data:
        return_str = return_str+child.data["INTBOOL"] + " "
      elif "INTINTINT" in child.data:
        return_str = return_str+child.data["INTINTINT"] + " "
    return return_str
    

current=ListProgram({"START": "start"})
current.production_inst("assign")
current.production_func("head")
current.production_arg("a")
current.production_var_assign("p")
print(current)
print(current.prog_to_string())


current.production_inst("assign_high")
print(current)
current.production_func_high("map")
print(current)
current.production_INTINT("(+1)")
print(current.prog_to_string())