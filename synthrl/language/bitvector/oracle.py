# from synthrl.language.bitvector.lang import ExprNode
import numpy as np
from synthrl.language.bitvector.lang import BitVectorLang
from synthrl.value.bitvector import BitVector
from synthrl.value.bitvector import BitVector16
from synthrl.value.bitvector import BitVector32
from synthrl.utils.trainutils import Dataset
from synthrl.language.bitvector.lang import VECTOR_LENGTH

def OracleSampler(sample_size=5, io_number=5):
    dataset = Dataset()
    for _ in range(sample_size):
        prog = generate_program(max_move=100)
        sample_ios = generate_io(prog, io_number=5, bit_length=16)
        dataset.add(prog, sample_ios)
    return dataset

#16bit range : [-(2^15),(2^15)-1]
#32bit range : [-(2^31), (2^31)-1]
def generate_io(prog, io_number=5, bit_length=16):
    ios = []
    for _ in range(io_number):
        low = -(2**bit_length)
        high = (2**bit_length)-1
        rand_val1 = np.random.randint(low,high) #integer
        rand_val2 = np.random.randint(low,high)
        output_val = prog.interprete((rand_val1,rand_val2))
        input_tuple = (rand_val1,rand_val2)
        ios.append((input_tuple,output_val))
    return ios

def generate_program(max_move=100):
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
    dataset = OracleSampler(5,5)
    for data in dataset.elements:
        data.program.pretty_print()
        print(data.program.tokenize())

