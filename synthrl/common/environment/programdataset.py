import random
from synthrl.common.environment.dataset import Dataset as Storage
import gc
from torch.utils.data import Dataset
from synthrl.common.language.bitvector import BitVectorLang

class ProgramDataset(Dataset):
  def __init__(self, dataset_paths = []):
    self.states = []
    #states : includes (partial_programs, idx of corresp. io_set)
    self.labels = []
    #labels : the right next seuqence of partial program

    self.storage = Storage("BitVectorLang") 
    for path in dataset_paths:
      Storage.join(self.storage, Storage.from_json(path))

    for idx, elt in enumerate(self.storage.elements):
      pgm_seq = (elt.oracle).sequence
      if "HOLE" in pgm_seq:
        pgm_seq.remove("HOLE")
      for i in range(len(pgm_seq)):
        # partial_pgm = BitVectorLang.tokens2prog(pgm_seq[:i])
        partial_pgm = pgm_seq[:i]
        ioset = idx
        self.states.append( ( partial_pgm , ioset) )
        self.labels.append(pgm_seq[i])
    
    gc.collect()
    assert len(self.states)==len(self.labels)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    partial_pgm = BitVectorLang.tokens2prog(self.states[index][0])
    io_idx = self.states[index][1]
    return (partial_pgm, self.storage.elements[io_idx].ioset), self.labels[index]


def iterate_minibatches(dataset, batch_size, shuffle=True):
  '''
  Source: https://stackoverflow.com/questions/38157972
  '''
  if shuffle:
    indices = list(range(len(dataset)))
    random.shuffle(indices)
  for start_idx in range(0, len(dataset), batch_size):
    if shuffle:
      excerpt = indices[start_idx:start_idx + batch_size]
      res = [dataset[idx] for idx in  excerpt]
      res = [list(t) for t in zip(*res)]
      yield res[0], res[1]
    else:
      excerpt = list(range(start_idx, start_idx + batch_size))
      res = [dataset[idx] for idx in  excerpt]
      res = [list(t) for t in zip(*res)]
      yield res[0], res[1]
      
    


