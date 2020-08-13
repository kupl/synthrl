from io import StringIO

from synthrl.common.environment.ioset import IOSet

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

  def to_json(self, json):
    pass

  @classmethod
  def from_json(cls, json):
    pass
