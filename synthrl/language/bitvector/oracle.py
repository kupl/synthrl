# from synthrl.language.bitvector.lang import ExprNode
import numpy as np
from synthrl.language.bitvector.lang import BitVectorLang

def OracleSampler(size=5, depth=5, io_number=5):
    pass

def generate_io(program, n_io=3):  
    pass

def generate_program(max_move):
    prog = BitVectorLang()
    space = prog.production_space()
    for i in range(max_move):
        action = np.random.choice(space)
        prog.production(action)
        space = prog.production_space()
        if len(space)==0:
            break
    return prog

if __name__ == '__main__':
    prog = generate_program(100)
    prog.pretty_print()
    for k in range(10):
        print("program ", k )
        prog = generate_program(100)
        prog.pretty_print()