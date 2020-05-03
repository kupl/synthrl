
# Exception to handle when the unexpected behavior of program is observed
class UnexpectedException(Exception):
  def __init__(self, *args, **kwargs):
    super(UnexpectedException, self).__init__(*args, **kwargs)
