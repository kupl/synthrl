import numpy as np

from synthrl.language.listlang.lang import T

from synthrl.utils.trainutils import Dataset
from synthrl.value import Integer
from synthrl.value import IntList
from synthrl.value.nonetype import NONE
from synthrl.language.listlang import ListLang
from synthrl.language.listlang.lang import InstNode
from synthrl.language.listlang.generate_io_samples import generate_IO_examples
from synthrl.language.listlang.generate_io_samples import compile
import time
from synthrl.language.abstract import Node
import os
import subprocess
import sys
import random
import json 

def encode_for_utile(program):
  representation = str()
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

            'pos'    : 'isPOS',
            'neg'    : 'isNEG', 
            'even' :  'isEVEN',
            'odd' :  'isODD',
            'min' : 'MIN',
            'max' : 'MAX',
            '+' : '+', 
            '*': '*'
        }
  global last_letter
  last_letter='a'
  if program.input_types ==[IntList]:
    var_dict['a_1'] = 'a'
    representation += "a <- [int]"
    last_letter ='a'
  elif program.input_types == [IntList,Integer]:
    var_dict['a_1'] = 'a'
    var_dict['a_2'] = 'b'
    representation += "a <- [int] | b <- int"
    last_letter ='b'
  else :
    var_dict['a_1'] = 'a'
    var_dict['a_2'] = 'b'
    representation += "a <- [int] | b <- [int]"
    last_letter ='b'
  for i, inst in enumerate(program.instructions) :
    if inst.data == 'nop':
      return representation
    last_letter = str(chr(ord(last_letter)+1))

    var_dict["x_"+ str(i+1)]=last_letter
    if(inst.data=='HOLE'):
      print("ERROR")
      print(program.pretty_print())
    option, n_vars, _ = InstNode.TOKENS[inst.data]

    func_name = (inst.data).upper()

    representation += " | " + last_letter + " <- " + func_name + " "
    if option == 'AUOP' and n_vars == 1:
      oper = op_dict[(inst.children['AUOP']).data]
      var = var_dict[(inst.children['VAR']).data]
      representation+= oper + " " + var
    elif option == 'BUOP' and n_vars == 1:
      oper = op_dict[(inst.children['BUOP']).data]
      var = var_dict[(inst.children['VAR']).data]   
      representation+= oper + " " + var   
    elif option == 'ABOP' and n_vars == 1:
      oper = op_dict[(inst.children['ABOP']).data]
      var = var_dict[(inst.children['VAR']).data]
      representation+= oper + " " + var
    elif option == 'ABOP' and n_vars == 2:
      oper = op_dict[(inst.children['ABOP']).data]
      var1 = var_dict[(inst.children['VAR1']).data]
      var2 = var_dict[(inst.children['VAR2']).data]
      representation+= oper + " " + var1 + " " + var2
    elif option == 'NOOPT' and n_vars == 1:
      var = var_dict[(inst.children['VAR']).data]
      representation+= var
    elif option == 'REQINT' and n_vars == 2:
      var1 = var_dict[(inst.children['VAR1']).data]
      var2 = var_dict[(inst.children['VAR2']).data]
      representation+=  var1 + " " + var2
  return representation

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
  return

def OracleSampler(size=5, depth=5, io_number = 5, io_set_len = 10, value_range = 512):
  input_types=[[IntList],[IntList,Integer],[IntList,IntList]]
  output_types=[IntList, Integer]
  dataset = Dataset()  
  # create size examples
  for j in range(size):
    input_type = random.choice(input_types)
    output_type = random.choice(output_types)
    program = ListLang(input_types=input_type, output_type=output_type)
    
    length = 0 
    action = None
    space = None
    while True:
      space = program.production_space
      if len(space)==0:
        break
      node = program.hole
      if isinstance(node, InstNode):
        length+=1
      if length < depth+1:
        if 'nop' in space: space.remove('nop')
      action = random.choice(space)
      if action == 'nop':
        break
      node.production(action)
    samples = IOSample(program,io_number, io_set_len, value_range)
    dataset.add(program, samples)
    if dataset.length() >size:
      return dataset
  return dataset


def IOSample(program, io_number = 5, io_set_len = 10, value_range = 512):
    source = encode_for_utile(program)
    print("---")
    print(source)
    cmd = 'python2 generate_io_samples_p2.py' + " -N " + str(io_number) + " -L " + str(io_set_len) + " -V " + str(value_range) 
    cmd= cmd.split()
    cmd.append(source)
    # print(cmd)
    samples = []
    try:
      cnt = 0
      proc = subprocess.check_output(cmd,universal_newlines=True)
      for line in proc.replace('\0', '').split('\n'):
        cnt+=1
        if cnt != (io_number+1):
          print(type(line))
          print(line)
          res = json.loads(line)
          _input = res[0]
          _output = res[1]
          sample = (_input,_output)
          samples.append(sample)
      time.sleep(1)
      return samples
    except subprocess.CalledProcessError:
      
      print("^^")
  


# dataset = OracleSampler()

