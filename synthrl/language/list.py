from typing import NewType
from lark import Lark

lips_grammar ="""
start: inst+

inst: VAR "<-" func ";"                     ->assign
    |VAR "<-" highfunc ";"                  ->assign_high
    |VAR "<-" "FIRST_INPUT" ";"             ->assign_first_input
    |VAR "<-" "SECOND_INPUT" ";"            ->assign_second_input
    | "if" bool "then" inst "else" inst ";" ->if_else
    | "end"                  ->end            

bool: "true" | "false" 
    | VAR "(>0)" | VAR "(<0)"
    | VAR "(%2==0)" | VAR "(%2==1)" 

VAR: "a" | "b" | "c" | "d" | "e" | "f" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "p"            
    | "x" | "y"                  
    | "o"

INTINT: "(+1)"   
    | "(-1)"    
    | "(*2)"    
    | "(/2)"    
    | "(*(-1))"  
    | "(**2)"  
    | "(*3)"    
    | "(/3)"    
    | "(*4)"    
    | "(/4)"    

INTBOOL: "(>0)" 
    | "(<0)"  
    | "(%2==0)"  
    | "(%2==1)"  

INTINTINT: "(+)"
    |"(-)"    
    |"(*)"     
    |"MIN"     
    |"MAX"    

highfunc: "MAP" INTINT VAR        ->map
    | "FILTER"   INTBOOL  VAR     ->filter
    | "COUNT"   INTBOOL VAR       ->count
    | "ZIPWITH" INTINTINT VAR VAR ->zipwith
    | "SCANL1" INTINTINT VAR VAR  ->scanl1

func: "HEAD" VAR        -> head
    | "LAST" VAR        -> last
    | "TAKE" VAR VAR    -> take
    | "DROP" VAR VAR    -> drop
    | "ACCESS" VAR VAR  -> access
    | "MINIMUM" VAR     -> minimum
    | "MAXIMUM" VAR     -> maximum
    | "REVERSE" VAR     -> reverse
    | "SORT" VAR        -> sort
    | "SUM" VAR         -> sum
    | "APPEND" VAR VAR  -> append
    | "CONS" VAR        -> cons
    | "INIT_BLANK"      ->init_blank
%import common.WS
%import common.LETTER
%ignore WS
"""


def run_lips(program,lips_grammar,inputs): ##input should be more at maximum 2.
  parser = Lark(lips_grammar)
  env=dict() #init environment

  ## bring in inputs onto envrionment##
  if len(inputs)==2:
    env['x']=inputs[0]
    env['y']=inputs[1]
  elif len(inputs)==1:
    env['x']=inputs[0]
  
  ## Setting Dict Just for Unit Test##
  #env['a']= 100
  #env['c']=[1,2,3,4,5]
  #env['d'] = 1
  #Desired Output is env['a']=1
  for inst in (parser.parse(program).children):
    if inst.data=="end":
      return env['o']
    else:
      env = run_insts(inst,env)   ##run each instructions

def run_insts(inst,env):
  if inst.data=="assign":
    [var,func] = inst.children
    #print(var[0])
    #Print target variable to assign
    env[var[0]] = run_function(func, env)
  elif inst.data=="assign_high":
    [var,func] = inst.children
    env[var[0]] = run_high_function(func, env)
  elif inst.data=="assign_first_input":
    env[inst.children[0]] = env['x'] 
  elif inst.data=="assign_second_input":
    env[inst.children[0]] = env['y'] 
  return env


def run_function(func,env):
  if func.data=="head":
    return (lambda xs: xs[0] if len(xs)>0 else Null) (env[func.children[0] ])
  elif func.data=="last":
    return (lambda xs: xs[-1] if len(xs)>0 else Null) (env[func.children[0] ])
  elif func.data=="take":
    return (lambda n, xs: xs[:n]) ((env[func.children[0]]), (env[func.children[1]]))
  elif func.data=="drop":
    return (lambda n, xs: xs[n:]) ((env[func.children[0]]), (env[func.children[1]]))
  elif func.data=="access":
    return (lambda n, xs: xs[n] if n>=0 and len(xs)>n else Null) ((env[func.children[0]]), (env[func.children[1]]))
  elif func.data=="minimum":
    return (lambda xs: min(xs) if len(xs)>0 else Null) (env[func.children[0]])
  elif func.data=="maximum":
    return (lambda xs: max(xs) if len(xs)>0 else Null)  (env[func.children[0]])
  elif func.data=="reverse":
    return (lambda xs: list(reversed(xs)))  (env[func.children[0]])
  elif func.data=="sort":
    return (lambda xs: sorted(xs))  (env[func.children[0]])
  elif func.data=="sum":
    return (lambda xs: sum(xs)) (env[func.children[0]])


lambdadic_int2int={
  "(+1)"    : lambda x: x+1        ,
  "(-1)"    : lambda x: x-1        ,
  "(*2)"    : lambda x: x*2        ,  
  "(/2)"    : lambda x: x/2        ,      
  "(*(-1))"  : lambda x: x*(-1)      ,  
  "(**2)"    : lambda x: x**2      ,
  "(*3)"     : lambda x: x*3        ,    
  "(/3)"     : lambda x: x/3        ,    
  "(*4)"    : lambda x: x*4        ,  
  "(/4)"    : lambda x: x/4  
}

lambdadic_int2bool={
  "(>0)"    : lambda x: x>0      ,
  "(<0)"    : lambda x: x<0      ,
  "(%2==0)"  : lambda x: (x%2==0)  ,  
  "(%2==1)"  : lambda x: (x%2==1)  
}

lambdadic_intint2int={
  "(+)"    : lambda x ,y: x+y        ,
  "(-)"    : lambda x,y: x-y        ,
  "(*)"    : lambda x,y: x*y        ,  
  "MAX"    : lambda x,y: max(x,y)      ,  
  "MIN"    : lambda x,y: min(x,y)
}



def run_high_function(func,env):
  if func.data=="map":
    return (lambda f, xs: [f(x) for x in xs]) (lambdadic_int2int[func.children[0]],  env[func.children[1]])
  elif func.data=="filter":
    return (lambda f, xs: [x for x in xs if f(x)]) (lambdadic_int2bool[func.children[0]],env[func.children[1]])
  elif func.data=="count":
    return (lambda f, xs: len([x for x in xs if f(x)])) (lambdadic_int2bool[func.children[0]],env[func.children[1]])
  elif func.data=="zipwith":
    return (lambda f, xs, ys: [f(x, y) for (x, y) in zip(xs, ys)]) (lambdadic_intint2int[func.children[0]],
                                    env[func.children[1]],
                                    env[func.children[2]])
  elif func.data=="scanl1":
    pass 

##Examples##
example_program="""
k<- FIRST_INPUT;
b<- SECOND_INPUT;
c<- SORT b;
d<- TAKE k c;
o<- SUM d;
end
"""

print(run_lips(example_program,lips_grammar,[2,[3,5,4,7,5] ]))
##Desired Output is 7

#class ListDSL(DSL):
#  def __init__(self):
#    pass
