import numpy as np

from synthrl.language.listlang import ListLang
from synthrl.language.listlang.lang import T
from synthrl.value import Integer
from synthrl.value import IntList
from synthrl.utils.trainutils import Dataset

def generate_program(input_types, output_type, length=T):

  prog = ListLang(input_types, output_type)

  space = prog.production_space
  while len(space) > 0:

    # if InstNode
    if 'nop' in space:
      if len(prog) < length:
        space.remove('nop')
      else:
        space = ['nop']

    action = np.random.choice(space)
    prog.production(action)
    space = prog.production_space

  return prog

def generate_io(program, n_io=3):

  input_types = program.input_types

  # generate n_io examples
  ios = []
  for _ in range(n_io):
    input_value = [ty.sample() for ty in input_types]
    output_value = program(input_value)
    ios.append((input_value, output_value))

  return ios

def OracleSampler(size=5, depth=5, io_number=5):

  input_types=[[IntList], [IntList, Integer], [IntList, IntList]]
  output_types=[IntList, Integer]
  
  # create size examples
  dataset = Dataset()  
  for _ in range(size):
    
    input_type = np.random.choice(input_types)
    output_type = np.random.choice(output_types)

    # sample program
    prog = generate_program(input_type, output_type, length=depth)

    # sample io
    samples = generate_io(prog, n_io=io_number)

    dataset.add(prog, samples)

  return dataset
