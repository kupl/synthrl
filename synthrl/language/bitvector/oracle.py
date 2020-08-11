# from synthrl.language.bitvector.lang import ExprNode
import numpy as np
from synthrl.language.bitvector.lang import BitVectorLang
from synthrl.value.bitvector import BitVector
from synthrl.value.bitvector import BitVector16
from synthrl.value.bitvector import BitVector32
from synthrl.utils.trainutils import Dataset
from synthrl.language.bitvector.lang import VECTOR_LENGTH

def prog2int(prog):
    #used for seeding procedure
    li = []
    num = 0
    tokenized = prog.tokenize()
    for token in tokenized:
        if not isinstance(token, str):
            token = str(token)
        li += [] + list(token)
    for w in li:
        if not isinstance(w, str):
            w=str(w)
        num += ord(w)
    return num

def OracleSampler(sample_size=5, io_number=5, seed=None):
    dataset = Dataset()
    for _ in range(sample_size):
        if seed is not None:
            seed+=i
        try:
            prog = generate_program(max_move=100,seed=seed)
            # prog.pretty_print()
            # print(prog.is_complete())
            if not prog.is_complete():
                raise ValueError
        except ValueError:
            continue
        sample_ios = generate_io(prog, io_number=io_number, bit_length=VECTOR_LENGTH, seed=seed)
        dataset.add(prog, sample_ios)
    return dataset

#16bit range : [-(2^15),(2^15)-1]
#32bit range : [-(2^31), (2^31)-1]
def generate_io(prog, io_number=5, bit_length=16,seed=None):
    ios = []
    for _ in range(io_number):
        if seed is not None:
            seed += prog2int(prog)
            np.random.seed(seed)
            seed += 1
        np.random.seed(seed)
        low = -(2**bit_length)
        high = (2**bit_length)-1
        rand_val1 = np.random.randint(low,high) #integer
        rand_val2 = np.random.randint(low,high)
        output_val = prog.interprete((rand_val1,rand_val2))
        input_tuple = (rand_val1,rand_val2)
        ios.append((input_tuple,output_val))
    return ios

def generate_program(max_move=100,seed=None):
    prog = BitVectorLang()
    space = prog.production_space()
    ind = 0
    while ind < max_move :
        if seed is not None:
            np.random.seed(seed)
            seed+=1
        np.random.seed(seed)
        action = np.random.choice(space)
        prog.production(action)
        space = prog.production_space()
        if len(space)==0:
            if prog.is_complete():
                break
            else:
                prog = BitVectorLang()
                space = prog.production_space()
                ind = 0
                continue
        ind+=1
    return prog

if __name__ == '__main__':
    dataset = OracleSampler(100,5,seed=None)
    # for data in dataset.elements:
    #     for io in data.ioset:
    #         print(io)
