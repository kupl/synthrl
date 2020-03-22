from synthrl.language.dsl import Tree
from synthrl.value import Integer
from synthrl.value import IntList
from synthrl.value.integer import ONE, TWO, THREE, FOUR

# wrapper class for program
# L -> P  # root
class ListLanguage(Tree):
  def __init__(self, *args, **kwargs):
    self.data = 'root'
    self.children = {'PGM': ProgramNode(parent=self)}
    self.parent = None

  def production_space(self):
    return self.children['PGM'].production_space()
  
  def production(self, rule=None):
    raise ValueError('ListLanguage should not have any hole.')

  def interprete(self, inputs=[]):
    mem = {v: [] for v in VarNode.var_space}
    for v, i in zip(VarNode.input_vars, inputs):
      mem[v] = Integer(i) if isinstance(i, int) else IntList(i)
    return self.children['PGM'].interprete(mem=mem)[VarNode.output_var].get_value()
  
  def pretty_print(self):
    self.children['PGM'].pretty_print()

# P -> I; P # seq
#    | eof  # eof
class ProgramNode(Tree):
  def production_space(self):
    if self.data == 'hole':
      return self, ['seq', 'eof']
    if self.data == 'seq':
      for key in ['INST', 'PGM']:
        node, space = self.children[key].production_space()
        if len(space) > 0:
          return node, space
      return self, []
    if self.data == 'eof':
      return self, []

  def production(self, rule=None):
    if rule == 'seq':
      self.data = 'seq'
      self.children = {
        'INST': InstNode(parent=self),
        'PGM': ProgramNode(parent=self)
      }
    elif rule == 'eof':
      self.data = 'eof'

  def interprete(self, mem={}):
    if self.data == 'seq':
      mem = self.children['INST'].interprete(mem)
      mem = self.children['PGM'].interprete(mem)
      return mem
    if self.data == 'eof':
      return mem

  def pretty_print(self):
    if self.data == 'hole':
      print('(HOLE)')
    elif self.data == 'seq':
      self.children['INST'].pretty_print()
      print(';')
      self.children['PGM'].pretty_print()
    elif self.data == 'eof':
      pass

# I -> V <- F # assign
class InstNode(Tree):
  def __init__(self, *args, **kwargs):
    super(InstNode, self).__init__(*args, **kwargs)
    # InstNode has only one production rule
    self.data = 'assign'
    self.children = {
      'VAR': VarNode(parent=self),
      'FUNC': FuncNode(parent=self)
    }

  def production_space(self):
    for key in ['FUNC', 'VAR']:
      node, space = self.children[key].production_space()
      if len(space) > 0:
        return node, space
    return self, []

  def production(self, rule=None):
    raise ValueError('InstNode should not have any hole.')
  
  def interprete(self, mem={}):
    if self.data == 'assign':
      value = self.children['FUNC'].interprete(mem)
      mem[self.children['VAR'].data] = value
      return mem

  def pretty_print(self):
    self.children['VAR'].pretty_print()
    print(' <- ', end='')
    self.children['FUNC'].pretty_print()

# V -> v0 | v1        # inputs
#    | v2 | ... | v18 # bounded variables
#    | v19            # output
class VarNode(Tree):
  VARIABLE_RANGE = 20
  var_space = ['v{}'.format(i) for i in range(20)]
  input_vars = ['v0', 'v1']
  output_var = 'v{}'.format(19)

  def production_space(self):
    if self.data == 'hole':
      return self, self.var_space
    else:
      return self, []
    
  def production(self, rule=None):
    if rule in self.var_space:
      self.data = rule

  def interprete(self, mem):
    if self.data in self.var_space:
      return mem[self.data]
    
  def pretty_print(self):
    print(self.data, end='')

# F -> MAP AUOP V       # map
#    | FILTER BUOP V    # filter
#    | COUNT BUOP V     # count
#    | ZIPWITH ABOP V V # zipwith
#    | SCANL ABOP V V   # scanl
#    | HEAD V           # head
#    | LAST V           # last
#    | MINIMUM V        # minimum
#    | MAXIMUM V        # maximum
#    | REVERSE V        # reverse
#    | SORT V           # sort
#    | SUM V            # sum
#    | TAKE V V         # take
#    | DROP V V         # drop
#    | ACCESS V V       # access
class FuncNode(Tree):
  auop_func = ['map']
  buop_func = ['filter', 'count']
  abop_func = ['zipwith', 'scanl']
  one_var_func = ['head', 'last', 'minimum', 'maximum', 'reverse', 'sort', 'sum']
  two_var_func = ['take', 'drop', 'access']

  def production_space(self):
    if self.data == 'hole':
      return self, self.auop_func + self.buop_func + self.abop_func + self.one_var_func + self.two_var_func
    keys = []
    if self.data in self.auop_func:
      keys = ['AUOP', 'VAR']
    if self.data in self.buop_func:
      keys = ['BUOP', 'VAR']
    if self.data in self.abop_func:
      keys = ['ABOP', 'VAR1', 'VAR2']
    if self.data in self.one_var_func:
      keys = ['VAR']
    if self.data in self.two_var_func:
      keys = ['VAR1', 'VAR2']
    for key in keys:
      node, space = self.children[key].production_space()
      if len(space) > 0:
        return node, space
    return self, []

  def production(self, rule=None):
    if rule in self.auop_func:
      self.data = rule
      self.children = {
        'AUOP': AUOPNode(parent=self),
        'VAR': VarNode(parent=self)
      }
    elif rule in self.buop_func:
      self.data = rule
      self.children = {
        'BUOP': BUOPNode(parent=self),
        'VAR': VarNode(parent=self)
      }
    elif rule in self.abop_func:
      self.data = rule
      self.children = {
        'ABOP': ABOPNode(parent=self),
        'VAR1': VarNode(parent=self),
        'VAR2': VarNode(parent=self)
      }
    elif rule in self.one_var_func:
      self.data = rule
      self.children = {
        'VAR': VarNode(parent=self)
      }
    elif rule in self.two_var_func:
      self.data = rule
      self.children = {
        'VAR1': VarNode(parent=self),
        'VAR2': VarNode(parent=self)
      }

  def interprete(self, mem={}):
    if self.data == 'map':
      f = self.children['AUOP'].interprete()
      xs = self.children['VAR'].interprete(mem)
      return [f(x) for x in xs]
    if self.data == 'filter':
      f = self.children['AUOP'].interprete()
      xs = self.children['VAR'].interprete(mem)
      return [x for x in xs if f(x)]
    if self.data == 'count':
      f = self.children['AUOP'].interprete()
      xs = self.children['VAR'].interprete(mem)
      return len([x for x in xs if f(x)])
    if self.data == 'zipwith':
      f = self.children['ABOP'].interprete()
      xs = self.children['VAR1'].interprete(mem)
      ys = self.children['VAR2'].interprete(mem)
      return [f(x, y) for x, y in zip(xs, ys)]
    if self.data == 'scanl':
      raise NotImplementedError
    if self.data == 'head':
      xs = self.children['VAR'].interprete(mem)
      return xs[0]
    if self.data == 'last':
      xs = self.children['VAR'].interprete(mem)
      return xs[-1]
    if self.data == 'minimum':
      xs = self.children['VAR'].interprete(mem)
      return min(xs)
    if self.data == 'maximum':
      xs = self.children['VAR'].interprete(mem)
      return max(xs)
    if self.data == 'reverse':
      xs = self.children['VAR'].interprete(mem)
      return list(reversed(xs))
    if self.data == 'sort':
      xs = self.children['VAR'].interprete(mem)
      return sorted(xs)
    if self.data == 'sum':
      xs = self.children['VAR'].interprete(mem)
      return sum(xs)
    if self.data == 'take':
      n = self.children['VAR1'].interprete(mem)
      xs = self.children['VAR2'].interprete(mem)
      return xs[:n]
    if self.data == 'drop':
      n = self.children['VAR1'].interprete(mem)
      xs = self.children['VAR2'].interprete(mem)
      return xs[n:]
    if self.data == 'access':
      n = self.children['VAR1'].interprete(mem)
      xs = self.children['VAR2'].interprete(mem)
      return xs[n]

  def pretty_print(self):
    if self.data == 'hole':
      print('(HOLE)', end='')
    elif self.data in self.auop_func:
      print(self.data.upper(), end=' ')
      self.children['AUOP'].pretty_print()
      print(end=' ')
      self.children['VAR'].pretty_print()
    elif self.data in self.buop_func:
      print(self.data.upper(), end=' ')
      self.children['BUOP'].pretty_print()
      print(end=' ')
      self.children['VAR'].pretty_print()
    elif self.data in self.abop_func:
      print(self.data.upper(), end=' ')
      self.children['ABOP'].pretty_print()
      print(end=' ')
      self.children['VAR1'].pretty_print()
      print(end=' ')
      self.children['VAR2'].pretty_print()
    elif self.data in self.one_var_func:
      print(self.data.upper(), end=' ')
      self.children['VAR'].pretty_print()
    elif self.data in self.two_var_func:
      print(self.data.upper(), end=' ')
      self.children['VAR1'].pretty_print()
      print(end=' ')
      self.children['VAR2'].pretty_print()

# AUOP -> +1 | -1 | *2 | /2 | *(-1) | **2 | *3 | /3 | *4 | /4
class AUOPNode(Tree):
  auop_space = ['+1', '-1', '*2', '/2', '*(-1)', '**2', '*3', '/3', '*4', '/4']

  def production_space(self):
    if self.data == 'hole':
      return self, self.auop_space
    else:
      return self, []

  def production(self, rule=None):
    if rule in self.auop_space:
      self.data = rule

  def interprete(self):
    if self.data == '+1':
      return lambda x: x + 1
    if self.data == '-1':
      return lambda x: x - 1
    if self.data == '*2':
      return lambda x: x * 2
    if self.data == '/2':
      return lambda x: x // 2
    if self.data == '*(-1)':
      return lambda x: -x
    if self.data == '**2':
      return lambda x: x * x
    if self.data == '*3':
      return lambda x: x * 3
    if self.data == '/3':
      return lambda x: x / 3
    if self.data == '*4':
      return lambda x: x * 4
    if self.data == '/4':
      return lambda x: x / 4
  
  def pretty_print(self):
    print('({})'.format(self.data), end='')


# BUOP -> >0 | <0 | %2==0 | %2==1
class BUOPNode(Tree):
  buop_space = ['>0', '<0', '%2==0', '%2==1']

  def production_space(self):
    if self.data == 'hole':
      return self, self.buop_space
    else:
      return self, []
    
  def production(self, rule=None):
    if rule in self.buop_space:
      self.data = rule

  def interprete(self):
    if self.data == '>0':
      return lambda x: x > 0
    if self.data == '<0':
      return lambda x: x < 0
    if self.data == '%2==0':
      return lambda x: (x % 2) == 0
    if self.data == '%2==1':
      return lambda x: (x % 2) == 1
    
  def pretty_print(self):
    print('({})'.format(self.data), end='')

# ABOP -> + | * | MIN | MAX
class ABOPNode(Tree):
  abop_space = '+', '*', 'MIN', 'MAX'

  def production_space(self):
    if self.data == 'hole':
      return self, self.abop_space
    else:
      return self, []

  def production(self, rule=None):
    if rule in self.abop_space:
      self.data = rule

  def interprete(self):
    if self.data == '+':
      return lambda x, y: x + y
    if self.data == '*':
      return lambda x, y: x * y
    if self.data == 'MIN':
      return lambda x, y: x if x < y else y
    if self.data == 'MAX':
      return lambda x, y: x if x > y else y

  def pretty_print(self):
    print('({})'.format(self.data), end='')
