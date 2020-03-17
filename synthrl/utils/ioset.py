from copy import deepcopy

class IOSet:
  def __init__(self, pairs=[]):
    if isinstance(pairs, IOSet):
      pairs = deepcopy(pairs.pairs)
    self.pairs = pairs

  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, idx):
    return self.pairs[idx]

  def __iter__(self):
    for pair in self.pairs:
      yield pair

  def copy(self):
    return IOSet(deepcopy(self.pairs))

  def append(self, pair=None):
    self.pairs.append(pair)
