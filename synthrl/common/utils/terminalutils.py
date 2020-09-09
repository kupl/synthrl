
class Color:
  PURPLE = '\033[95m'
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  RED = '\033[91m'
  END = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

  @classmethod
  def disable(cls):
    cls.PURPLE = ''
    cls.BLUE = ''
    cls.GREEN = ''
    cls.YELLOW = ''
    cls.RED = ''
    cls.END = ''
    cls.BOLD = ''
    cls.UNDERLINE = ''

def mktable(rows, header=None, align='<', index=False):
  table = [(['Index'] if index else []) + header]
  col_length = list(map(len, table[0]))
  for idx, row in enumerate(rows):
    row = ([idx] if index else []) + list(row)
    row = list(map(repr, row))
    col_length = [max(old, new) for old, new in zip(col_length, map(len, row))]
    table.append(row)
  
  if not isinstance(align, (list, tuple)):
    align = [align] * len(col_length)
  header, content = table[0], table[1:]
  string = ' '.join((f'{{{":" + align + str(length) + "s"}}}'.format(name) for length, align, name in zip(col_length, align, header)))
  string += '\n'
  string += ' '.join(('-' * col for col in col_length))
  string += '\n'
  for row in content:
    string += ' '.join((f'{{{":" + align + str(length) + "s"}}}'.format(data) for length, align, data in zip(col_length, align, row)))
    string += '\n'
  return string
