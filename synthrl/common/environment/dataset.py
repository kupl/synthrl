from ast import literal_eval
from io import StringIO
import json
import numpy as np

from synthrl.common.environment.ioset import IOSet
import synthrl.common.language as language_module

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

  def __init__(self, language):
    self.language = language
    self.elements = []

  def __getitem__(self, idx):
    return self.elements[idx]

  def add(self, program, ioset):
    self.elements.append(Element(program, ioset))
  
  def join(self, other):
    self.elements += other.elements
    
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

  def __len__(self):
    return self.length

  @property
  def length(self):
    return len(self.elements)

  def to_json(self, file, indent=2):
	  # file: path for saving data.json
    # save self.elements as json    

    dataset = {
      'language': self.language,
      'data': [],
    }
    
    for oracle, ioset in self.elements:

      # Get program as string.
      with StringIO() as stream:
        oracle.pretty_print(file=stream)
        oracle = stream.getvalue().strip()

      # Get ioset as string.
      ioset = [ str(  ((int(io[0][0]),int(io[0][1])) ,  int(io[1]) ) )    for io in ioset]
      
      # Add to dataset.
      dataset['data'].append({
        'oracle': oracle,
        'ioset': ioset
      })
    
    with open(file, 'w', encoding='utf-8') as f:
      json.dump(dataset, f, indent=indent)
  
  @classmethod
  def from_json(cls, file, language=None):
	  # file: path to data.json

    with open(file, 'r', encoding='utf-8') as f:
      dataset = json.load(f)
    
    res = cls(dataset['language'])
    if not language:
      language = getattr(language_module, dataset['language'])

    for element in dataset['data']:
      oracle = language.parse(element['oracle'])
      ioset = [literal_eval(io) for io in element['ioset']]
      ioset = [([language.VALUE(i) for i in inputs], language.VALUE(output))for inputs, output in ioset]
      res.add(oracle, ioset)

    return res

  def to_txt(self, file):
    pass

  def from_txt(self, file):
    pass
