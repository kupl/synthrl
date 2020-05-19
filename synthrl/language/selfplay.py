from synthrl.language.dsl import Tree
from synthrl.language.listlang import ListLanguage
from synthrl.language.listlang import VarNode
from synthrl.language.listlang import InstNode
from synthrl.language.listlang import FuncNode
from synthrl.language.dsl import UndefinedSemantics
import numpy as np
from synthrl.value import Integer
from synthrl.value import IntList
from synthrl.value import NoneType
import random
import subprocess
import os

class SelfPlay:
    def __init__(self):
        pass
    def generate_indiv_oracle(depth,input_type, output_type):
        action = None
        program = ListLanguage(input_types=input_type, output_type=output_type)
        length = 0 #as appearance count of 'seq' action
        used_variable=[]

        while True:
            node, space = program.production_space()
            if space ==["seq","return"] and length < depth:
                if "return" in space: space.remove("return")
                length += 1
            elif space ==["seq","return"] and length == depth:
                node.production("return")
                return_node, return_space = program.production_space()
                if len(return_space)==0 :
                    #reset sampling
                    program  = ListLanguage(input_types=input_type, output_type=output_type)
                    action = None
                    length = 0
                else:
                    if ('v'+str(length-1)) in return_space:
                        return_node.production('v'+str(length-1))
                        return program
                    else:
                        #reset sampling
                        program  = ListLanguage(input_types=input_type, output_type=output_type)
                        action = None
                        length = 0
                    # return_node.production('v' + str(length-1))
            #Node choice in incremental order
            if type(node)==VarNode and type(node.parent)==InstNode:
                action= 'v' + str(length-1)
                print(action)
            else: action = np.random.choice(space)
            node.production(action)

    @classmethod 
    def generate_oracles(self, depth, size):
        dataset = []
        possible_input_types=[(list,None),(list,int),(list,list) ]
        possible_output_types=[list,int]
        for i in range(size):
            input_type = random.choice(possible_input_types)
            output_type = random.choice(possible_output_types)
            program = self.generate_indiv_oracle(depth,input_type,output_type)
            dataset.append(program)
        return dataset 

    def io_sampling(program):
        inputs = []
        print(program.input_types)
        try_count = 0
        while True and try_count<=1000:
            for type in program.input_types:
                if type==Integer:
                    inputs.append(Integer.sample())
                elif type==IntList:
                    inputs.append(IntList.sample())
                elif type==None:
                    pass
            try:
                output=program.interprete(inputs)
                return inputs,output
            except UndefinedSemantics as e:
                try_count +=1
                if try_count >100:
                    print(e)
                    return None, None

    def encode_to_representation(program):
        #Input: full program without holes.
        representation = str() #as string
        var_dict = {}
        op_dict={
            '+1'    : 'INC',
            '-1'    : 'DEC',
            '*2'    : 'SHL',
            '/2'    : 'SHR', 
            '*(-1)' : 'doNEG',
            '**2'   : 'SQR',
            '*3'    : 'MUL3',
            '/3'    : 'DIV3',
            '*4'    : 'MUL4',
            '/4'    : 'DIV4',

            '>0'    : 'isPOS',
            '<0'    : 'isNEG', 
            '%2==0' :  'isODD',
            '%2==1' :  'isEVEN'
        }

        global last_letter
        last_letter='a'
        if program.input_types ==[IntList,NoneType]:
            var_dict['a0'] = 'a'
            representation += "a <- [int]"
            last_letter ='a'
        elif program.input_types == [IntList,Integer]:
            var_dict['a0'] = 'a'
            var_dict['a1'] = 'b'
            representation += "a <- [int] | b <- int"
            last_letter ='b'
        else :
            var_dict['a0'] = 'a'
            var_dict['a1'] = 'b'
            representation += "a <- [int] | b <- [int]"
            last_letter ='b'
        prog_node = program.children['PGM']
        while True:
             if prog_node.data=="seq":
                inst = prog_node.children['INST']
                # inst.pretty_print()
                
                last_letter = str(chr(ord(last_letter)+1) )
                var_dict[inst.children['VAR'].data] = last_letter
                func_name = ((inst.children["FUNC"]).data).upper()
                representation += " | " + last_letter + " <- " + func_name + " "

                func = inst.children["FUNC"]
                if func.data in FuncNode.AUOP_FUNC_RETL:
                    representation +=  op_dict[func.children["AUOP"].data] + " "
                    representation += var_dict[func.children['VAR'].data] 
                elif func.data in FuncNode.BUOP_FUNC_RETL + FuncNode.BUOP_FUNC_RETI:
                    representation += op_dict[func.children["BUOP"].data] + " "
                    representation += var_dict[func.children['VAR'].data]
                elif func.data in FuncNode.ABOP1_FUNC_RETL:
                    representation += func.children["ABOP"].data + " "
                    representation += var_dict[func.children['VAR'].data] 
                elif func.data in FuncNode.ABOP2_FUNC_RETL:
                    representation += func.children["ABOP"].data + " "
                    representation += var_dict[func.children['VAR1'].data] + " "
                    representation += var_dict[func.children['VAR2'].data] 
                elif func.data in FuncNode.ONE_VAR_FUNC_RETL + FuncNode.ONE_VAR_FUNC_RETI:
                    representation += var_dict[func.children['VAR'].data]
                elif func.data in FuncNode.TWO_VAR_FUNC_RETL + FuncNode.TWO_VAR_FUNC_RETI:
                    representation += var_dict[func.children['VAR1'].data] + " "
                    representation += var_dict[func.children['VAR2'].data] 
                prog_node=prog_node.children['PGM']
             elif prog_node.data=="return":
                 print(representation)
                 return representation


    # def io_query(inputs,program):
    #     try:
    #         output=program.interprete(inputs)
    #     except UndefinedSemantics:
    #         print("semantics error!!!!")

# program = ListLanguage(input_types=(list, int), output_type=list)
dataset = SelfPlay.generate_oracles(4,1)
print("--**Generated Oracle**--")
for data in dataset:
    print(data.input_types)
    data.pretty_print()
    line = 'python2 generate_io_samples.py' + " " + '\"'  +  SelfPlay.encode_to_representation(data) +'\"'
    aaaa=os.popen(line)
    print(aaaa.read())
