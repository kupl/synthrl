from synthrl.utils.trainutils import Dataset

def OracleSampler(size=1000):

  # create dataset
  dataset = Dataset()

  # create size examples
  for _ in range(size):

    # create oracle and ioset
    # TODO
    program = None
    ioset = None

    # add to dataset
    dataset.add(program, ioset)

  return dataset
