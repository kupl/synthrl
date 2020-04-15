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

        while action!="return":
            node, space = program.production_space()
            if space ==["seq","return"] and length < depth:
                if "return" in space: space.remove("return")
                length += 1
            elif space ==["seq","return"] and length == depth:
                node.production("return")
                return_node, return_space = program.production_space()
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
        for i in range(size):
            dataset.append(self.generate_indiv_oracle(depth))
        return dataset
    
    def preprocess(dataset):
        pass

    def io_query():
        pass



dataset = SelfPlay.generate_oracles(4,10)
for data in dataset:
    print("--**Generated Oracle**--")
    data.pretty_print()