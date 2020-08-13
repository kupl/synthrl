import json
from ast import literal_eval
from io import StringIO

from synthrl.language.bitvector.lang import BitVectorLang
from synthrl.language.listlang.lang import ListLang
from synthrl.language.bitvector import ExprNode
from contextlib import redirect_stdout

# list of input and output pairs
class IOSet:

  __slots__ = ['pairs']

  def __init__(self, pairs):
    self.pairs = pairs

  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, idx):
    return self.pairs[idx]

  def append(self, pair):
    self.pairs.append(pair)

# Tuple of (program, ioset), each element of Dataset
class Element:

  __slots__ = ['_program', 'ioset']

  def __init__(self, program, ioset):
    
    self._program = program
    self.ioset = IOSet(ioset)

  @property
  def program(self):

    # to make program immutable
    # return self._program.copy()
    return self._program

# List of (program, ioset)
class Dataset:

  __slots__ = ['elements']

  def __init__(self, elements=[]):
    # elements: A list of pairs of (program, [(inputs, output)]).
    
    # Wrap with wrapper
    self.elements = [Element(*e) for e in elements]

  def __getitem__(self, idx):
    # idx: Index to search.
    
    # return idx'th element
    return self.elements[idx]

  def add(self, program, ioset):
    # program: A new program. Must be callable object.
    # io     : A list of tuples of inputs and outputs corresponding to the given program.
    
    # create Element object
    elem = Element(program, ioset)
    self.elements.append(elem)

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

  def __repr__(self):
    
    # make string using StingIO
    stream = StringIO()

    # for each element
    for element in self.elements:

      # print program
      print('--program--', file=stream)
      element.program.pretty_print(file=stream)
      
      # print io pairs
      print('--io set--', file=stream)
      for pair in element.ioset:
        print(pair, file=stream)

      print(file=stream)

    # get string and return
    string = stream.getvalue()
    stream.close()
    return string

  def __str__(self):
    return repr(self)
  
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
        elif isinstance(program, BitVectorLang):
          pgm_type = "Bitvector"
          with StringIO() as buf, redirect_stdout(buf):
            program.pretty_print(file=buf)
            program = buf.getvalue()
          program = replace_ops(program)
          program = program.replace('\n', '').strip()
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

  def __repr__(self):
    
    # make string using StingIO
    stream = StringIO()

    # for each element
    for element in self.elements:

      # print program
      print('--program--', file=stream)
      element.program.pretty_print(file=stream)
      
      # print io pairs
      print('--io set--', file=stream)
      for pair in element.ioset:
        print(pair, file=stream)

      print(file=stream)

    # get string and return
    string = stream.getvalue()
    stream.close()
    return string
