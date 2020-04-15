from synthrl.language.dsl import Tree
from synthrl.language.listlang import ListLanguage
import numpy as np


class SelfPlay:
    def __init__(self):

        pass
    def generate_indiv_oracle(depth,dsl="list"):
        action = None
        program = ListLanguage()
        length = 0 #as appearance count of 'seq' action

        while action!="eof":
            node, space = program.production_space()
            if space ==["seq","eof"] and length < depth:
                if "eof" in space: space.remove("eof")
                length += 1
            elif space ==["seq","eof"] and length == depth:
                node.production("eof")
                # print("----Generated Oracle-----")
                # program.pretty_print()
                # print("----****************-----")
                return program
            #print("Current Action Space:" ,space)
            action = np.random.choice(space)
            #print("Current Action Choice: ", action)
            node.production(action)
            #print("-------------")
            # program.pretty_print()
    
    @classmethod 
    def generate_oracles(self, depth, size):
        dataset = []
        for i in range(size):
            dataset.append(self.generate_indiv_oracle(depth))
        return dataset



dataset = SelfPlay.generate_oracles(4,10)
for data in dataset:
    print("--**Generated Oracle**--")
    data.pretty_print()