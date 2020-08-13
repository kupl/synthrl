from ast import literal_eval
from contextlib import redirect_stdout
from io import StringIO
import json

from synthrl.common.environment.ioset import IOSet
from synthrl.language.bitvector.lang import BitVectorLang
from synthrl.language.listlang.lang import ListLang
from synthrl.language.bitvector import ExprNode

class Element:

  def __init__(self, oracle, ioset):
    self.oracle = oracle
    self.ioset = IOSet(ioset)

  def __getitem__(self, idx):
    if idx == 0:
      return self.oracle
    elif idx == 1:
      return self.ioset
    else:
      raise IndexError('Index 0 for oracle, and index 1 for ioset.')

class Dataset:

  def __init__(self):
    self.elements = []

  def __getitem__(self, idx):
    return self.elements[idx]

  def add(self, program, ioset):
    self.elements.append(Element(program, ioset))

  def __repr__(self):

    stream = StringIO()

    for pgm, ioset in self.elements:
      print('--program--', file=stream)
      pgm.pretty_print(file=stream)
      
      print('--io set--', file=stream)
      for pair in ioset:
        print(pair, file=stream)

      print(file=stream)

    string = stream.getvalue()
    stream.close()
    return string

  def __str__(self):
    return repr(self)

  @property
  def length(self):
    return len(self.elements)

  def to_json(self, file):
	  # file: path for saving data.json
    # save self.elements as json    
    
    def replace_ops(program):
      program = program.replace("ARITH-NEG", "-")
      program = program.replace("NEG", "¬")
      return program
    
    with open(file, 'w', encoding="utf-8") as f:
      print("{", file=f)
      print("  \"data\": [", file=f)
      # for each program & ioset
      for e in self.elements:
        program = e.program
        if isinstance(program, ListLang):
          pgm_type = "List"
          with StringIO() as buf, redirect_stdout(buf):
            program.pretty_print(file=buf)
            program = buf.getvalue()
          program = program.replace('\n', ' ').strip()
        elif isinstance(program, ExprNode):
          pgm_type = "Bitvector"
          with StringIO() as buf, redirect_stdout(buf):
            program.pretty_print(file=buf)
            program = buf.getvalue()
          program = replace_ops(program)
        else:
          raise ValueError("Invalid program tree is given")
        ioset = e.ioset
        print("    {", file=f)
        print("      \"program_type\": \"{}\",".format(pgm_type), file=f)
        print("      \"pgm\": \"{}\",".format(program), file=f)
        print("      \"io\": [", file=f)
        print("        {}".format(",\n        ".join(
          ["\"{}\"".format(a) for a in ioset])), file=f)
        print("      ]", file=f)
        print("    }" if e == self.elements[-1] else "    },", file=f)
      print("  ]", file=f)
      print("}", file=f)
  
  @classmethod
  def from_json(self, file):
	  # file: path to data.json
    res = Dataset()
    with open(file, 'r', encoding="utf-8") as f:
      raw_json = json.load(f)
      for e in raw_json['data']:
        pgm_type = e['program_type']
        program = e['pgm']
        if pgm_type == "List":
          parser = ListLang.parse
        elif pgm_type == "Bitvector":
          parser = BitVectorLang.parse
        else:
          return SyntaxError("Invalid program type {} for {}".format(pgm_type, program))
        program = parser(program)
        ioset = [literal_eval(a) for a in e['io']]
        res.add(program, ioset)
    return res
