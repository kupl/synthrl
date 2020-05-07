from io import StringIO

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
