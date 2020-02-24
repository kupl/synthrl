from synthrl.language.dsl import DSL
from lark import Lark

lips_grammar ="""
start: inst+

inst: VAR "<-" func ";"						->assign
	| "if" bool "then" inst "else" inst ";"	->if_else
	| "end"									->end						

bool: "true" | "false" 
	| VAR "(>0)" | VAR "(<0)"
	| VAR "(%2==0)" | VAR "(%2==1)" 

VAR: "a"									
	| "b" 									
	| "c" 									
	| "d" | "e" | "f" 						
	| "x" 									
	| "y"									
	| "o"									


func: "HEAD" VAR		-> head
	| "LAST" VAR		-> last
	| "TAKE" VAR VAR	-> take
	| "DROP" VAR VAR	-> drop
	| "ACCESS" VAR VAR	-> access
	| "MINIMUM" VAR		-> minimum
	| "MAXIMUM" VAR		-> maximum
	| "REVERSE" VAR		-> reverse
	| "SORT" VAR		-> sort
	| "SUM" VAR			-> sum
	| "APPEND" VAR VAR	-> append
	| "CONS" VAR		-> cons
	| "INIT_BLANK"		->init_blank


%import common.WS
%import common.LETTER
%ignore WS

"""
program="""
a <- TAKE d c;
end
"""
parser = Lark(lips_grammar)

print(parser.parse(program).children)
print("-----")


def run_lips(program,lips_grammar,input): ##input should be more at maximum 2.
	parser = Lark(lips_grammar)
	env=dict() #init environment

	######
	#Setting Dict Just for Unit Test#
	env['a']= 100
	env['c']=[1,2,3,4,5]
	env['d'] = 1
	#Desired Output is env['a']=1
	for inst in (parser.parse(program).children):
		if inst.data=="end":
			print(env)
			return env
		else:
			env = run_insts(inst,env) 	##run each instructions
	pass

def run_insts(inst,env):
	if inst.data=="assign":
		[var,func] = inst.children
		print(var[0]) ##print target variable to assign
		##Getting token value with print
		env[var[0]] = run_function(func, env)
	return env

def run_function(func,env):

	##Single Argument Functions
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

class ListDSL(DSL):
  def __init__(self):
    pass
