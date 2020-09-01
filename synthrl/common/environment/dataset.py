from ast import literal_eval
from contextlib import redirect_stdout
from io import StringIO
import json
import gc
from torch.utils.data import Dataset
from synthrl.common.environment.ioset import IOSet
from synthrl.common.language.bitvector.lang import BitVectorLang

import synthrl.common.language as language

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

class Storage:

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
      ioset = [str(io) for io in ioset]
      
      # Add to dataset.
      dataset['data'].append({
        'oracle': oracle,
        'ioset': ioset
      })
    
    with open(file, 'w', encoding='utf-8') as f:
      json.dump(dataset, f, indent=indent)
  
  @classmethod
  def from_json(cls, file):
	  # file: path to data.json

    with open(file, 'r', encoding='utf-8') as f:
      dataset = json.load(f)
    
    res = cls(dataset['language'])
    language_class = getattr(language, dataset['language'])

    for element in dataset['data']:
      oracle = language_class.parse(element['oracle'])
      ioset = [literal_eval(io) for io in element['ioset']]
      packed_ioset = IOSet([])
      for io in ioset:
        (inputs, output) = io
        (input1, input2) = inputs
        # pylint: disable=too-many-function-args
        inputs = (BitVectorLang.BITVECTOR(input1), BitVectorLang.BITVECTOR(input2))
        output = BitVectorLang.BITVECTOR(output)
        packed_ioset.add(inputs,output)

      res.add(oracle, packed_ioset)

    return res

  def to_txt(self, file):
    pass

  def from_txt(self, file):
    pass


class ProgramDataset(Dataset):
  def __init__(self, dataset_paths = []):
    self.states = []
    #states : includes (partial_programs, idx of corresp. io_set)
    self.labels = []
    #labels : the right next seuqence of partial program

    self.storage = Storage("BitVectorLang") 
    for path in dataset_paths:
      Storage.join(self.storage, Storage.from_json(path))

    for elt in self.storage.elements:
      pgm_seq = (elt.oracle).sequence
      for i in range(len(pgm_seq)):
        partial_pgm = BitVectorLang.tokens2prog(pgm_seq[:i])
        self.states.append( ( partial_pgm , elt.ioset) )
        self.labels.append(pgm_seq[i])
    gc.collect()
  
  def __len__(self):
    return len(self.labels)

  # def __getitem__(self, index):
  #   partial_pgm = BitVectorLang.tokens2prog(self.states[index][0])
  #   io_idx = self.states[index][1]
  #   return (partial_pgm, self.storage.elements[io_idx].ioset ), self.labels[index]
    
if __name__ == '__main__':
  # paths = ["../dataset/train/train_dataset_uptolv01.json",
  #           "../dataset/train/train_dataset_uptolv2.json",
  #           "../dataset/train/train_dataset_uptolv3.json",
  #           "../dataset/train/train_dataset_uptolv4.json",
  #           "../dataset/train/train_dataset_uptolv5.json" ]
  
  paths = ["../dataset/train/train_dataset_uptolv01.json" ] 
  dataset = ProgramDataset(dataset_paths=paths)
  assert len(dataset.states) == len(dataset.labels)


  # for state, label in zip(dataset.states ,dataset.labels):
  #   print(state, label ,end="\n")
  #   BitVectorLang.tokens2prog(state[0]).pretty_print()

  # paths = ["../dataset/train/temp_io.json"]
  # dataset = ProgramDataset(dataset_paths=paths)
  # (pp, io), label = dataset[12]
  # pp.pretty_print()
  # print(io)
  # print(label)


