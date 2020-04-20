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
        while True:
            node, space = program.production_space()
            #처음 선택되는 function이 2-argument function이면, 필연적으로 Hole이 생기고 이에 대한 대처
            if length==1:
                try:
                    space.remove('take')
                    space.remove('drop')
                    space.remove('access')
                except ValueError:
                    pass  
            if space ==["seq","return"] and length < depth:
                if "return" in space: space.remove("return")
                length += 1
            elif space ==["seq","return"] and length == depth:
                node.production("return")
                return_node, return_space = program.production_space()
                if len(return_space)==0 :
                    #print("the space is empty..")
                    #program.pretty_print()
                    #print(program.output_type)
                    ##tryagain
                    program  = ListLanguage(input_types=input_type, output_type=output_type)
                    action = None
                    length = 0
                else:
                    return_node.production(np.random.choice(return_space))
                    return program
            action = np.random.choice(space)
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


    def input_sampling(program):
        inputs = []
        print(program.input_types)
        for type in program.input_types:
            if type==Integer:
                inputs.append(Integer.sample())
            elif type==IntList:
                inputs.append(IntList.sample())
            elif type==None:
                pass
        return inputs

    def io_query(inputs,program):
        output=program.interprete(inputs)
        print(output)

# program = ListLanguage(input_types=(list, int), output_type=list)
dataset = SelfPlay.generate_oracles(4,10)
print("--**Generated Oracle**--")
for data in dataset:
    data.pretty_print()
    # inputs=SelfPlay.input_sampling(data)
    # print(inputs)
    # SelfPlay.io_query(inputs,data)