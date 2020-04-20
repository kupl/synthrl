from synthrl.language.dsl import Tree
from synthrl.language.listlang import ListLanguage
import numpy as np
from synthrl.value import Integer
from synthrl.value import IntList
from synthrl.value import NoneType
import random

class SelfPlay:
    def __init__(self):
        pass
    def generate_indiv_oracle(depth,input_type, output_type):
        action = None
        program = ListLanguage(input_types=input_type, output_type=output_type)
        length = 0 #as appearance count of 'seq' action
        while action!="return":
            node, space = program.production_space()
            if space ==["seq","return"] and length < depth:
                if "return" in space: space.remove("return")
                length += 1
            elif space ==["seq","return"] and length == depth:
                node.production("return")
                return_node, return_space = program.production_space()
                if len(return_space)==0 :
                    print("the space is empty..")
                    program.pretty_print()
                    print(program.output_type)
                    ##re init
                    prgoram = ListLanguage(input_types=input_type, output_type=output_type)
                    action = None
                    length = 0
                else:
                    return_node.production(np.random.choice(return_space))
                    # print("----Generated Oracle-----")
                    # program.pretty_print()
                    # print("----****************-----")
                    return program
            # print("Current Action Space:" ,space)
            action = np.random.choice(space)
            # print("Current Action Choice: ", action)
            node.production(action)
            #print("-------------")
            # program.pretty_print()
    
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
    
    def io_query(program):
        inputs = []
        for type in program.input_types:
            if type==int:
                input.append(random.randint(a, b))
            elif type==list:
                listlen = random.randint(1,10)
                elm_min = 0
                elm_max = 99
                input.append([random.randint(elm_min, elm_max) for i in range(listlen)])
            elif type==None:
                pass


# program = ListLanguage(input_types=(list, int), output_type=list)

dataset = SelfPlay.generate_oracles(4,10)
for data in dataset:
    print("--**Generated Oracle**--")
    data.pretty_print()
    print(data.input_types)
    SelfPlay.io_query(data)