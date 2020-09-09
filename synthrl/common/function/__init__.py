from synthrl.common.function.rand import RandomFunction
from synthrl.common.function.rnn import RNNFunction

def TrainedRNNFunction(path):
  return RNNFunction.load(path)
