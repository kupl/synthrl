from synthrl.language.dsl import DSL
from lark import Lark

lips_grammar ="""
start: p

p	: var_int "<-" func ";" p
	| var_list "<-" func ";" p
	| "if" bool "then" p "else" ";" p
	| "end"

bool: "true" | "false" 
	| var_int "(>0)" | var_int "(<0)"
	| var_int "(%2==0)" | var_int "(%2==1)" 

INPUT : "FIRST" | "SECOND"

var_int :  VAR  | INPUT
var_list : VAR | INPUT

func: "HEAD" var_list
	| "LAST" var_list
	| "TAKE" var_int var_list
	| "DROP" var_int var_list
	| "ACCESS" var_int var_list
	| "MINIMUM" var_list
	| "MAXIMUM" var_list
	| "REVERSE" var_list
	| "SORT" var_list
	| "SUM" var_list
	| "APPEND" var_list var_list
	| "CONS" var_list
	| "INIT_BLANK"

VAR: LETTER | "OUTPUT"


%import common.WS
%import common.LETTER
%ignore WS

"""

parser = Lark(lips_grammar)  

text="""
a <- HEAD FIRST;
end
"""

print(parser.parse(text).pretty())



class ListDSL(DSL):
  def __init__(self):
    pass
####asdf