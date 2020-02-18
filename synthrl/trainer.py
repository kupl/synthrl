class Trainer:
  def __init__(self, synth=None, alt=None):
    self.synth = synth
    self.alt = alt
  
  def fit(self, data=None, epochs=1):
    pass

  def get_agents(self):
    return (self.synth, self.alt)
