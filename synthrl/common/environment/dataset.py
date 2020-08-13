from synthrl.common.environment.ioset import IOSet

class Element:

  def __init__(self, oracle, ioset):
    self.oracle = oracle
    self.ioset = IOSet(ioset)

class Dataset:

  def __init__(self, oracle, ioset):
    self.oracle = oracle
    self.ioset = IOSet(ioset)

  def to_json(self, json):
    pass

  @classmethod
  def from_json(cls, json):
    pass
