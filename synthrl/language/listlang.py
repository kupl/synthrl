from synthrl.language.dsl import Tree
from synthrl.language.dsl import UndefinedSemantics
from synthrl.utils import classproperty
from synthrl.value import Integer
from synthrl.value import IntList
from synthrl.value.integer import ZERO, ONE, TWO, THREE, FOUR

# wrapper class for program
# L -> P  # root
class ListLanguage(Tree):
  def __init__(self, *args, **kwargs):
    self.data = 'root'
    self.children = {'PGM': ProgramNode(parent=self)}
    self.parent = None

  def production_space(self):
    node, space, _ = self.children['PGM'].production_space(used_vars=set(VarNode.INPUT_VARS))
    return node, space
  
  def production(self, rule=None):
    raise ValueError('ListLanguage should not have any hole.')

  def interprete(self, inputs=[]):
    mem = {v: IntList() for v in VarNode.VAR_SPACE}
    for v, i in zip(VarNode.INPUT_VARS, inputs):
      mem[v] = Integer(i) if isinstance(i, Integer) or isinstance(i, int) else IntList(i)
    return self.children['PGM'].interprete(mem=mem)
  
  def pretty_print(self, file=None):
    self.children['PGM'].pretty_print(file=file)

  def is_hole(self):
    return self.children['PGM'].is_hole()

  @classproperty
  @classmethod
  def tokens(cls):
    return ProgramNode.tokens + InstNode.tokens + VarNode.tokens + FuncNode.tokens + AUOPNode.tokens + BUOPNode.tokens + ABOPNode.tokens

# P -> I; P       # seq
#    | return V;  # return
class ProgramNode(Tree):
  def production_space(self, used_vars=set()):
    if self.data == 'hole':
      return self, ['seq', 'return'], used_vars
    if self.data == 'seq':
      for key in ['INST', 'PGM']:
        node, space, used_vars = self.children[key].production_space(used_vars)
        if len(space) > 0:
          return node, space, used_vars
      return self, [], used_vars
    if self.data == 'return':
      node, space, used_vars = self.children['VAR'].production_space(used_vars)
      if len(space) > 0:
        return node, space, used_vars
      return self, [], used_vars

  def production(self, rule=None):
    if rule == 'seq':
      self.data = 'seq'
      self.children = {
        'INST': InstNode(parent=self),
        'PGM': ProgramNode(parent=self)
      }
    elif rule == 'return':
      self.data = 'return'
      self.children = {
        'VAR': VarNode(parent=self)
      }

  def interprete(self, mem={}):
    if self.data == 'seq':
      mem = self.children['INST'].interprete(mem)
      mem = self.children['PGM'].interprete(mem)
      return mem
    if self.data == 'return':
      return self.children['VAR'].interprete(mem)

  def pretty_print(self, file=None):
    if self.data == 'hole':
      print('(HOLE)', file=file)
    elif self.data == 'seq':
      self.children['INST'].pretty_print(file=file)
      print(';', file=file)
      self.children['PGM'].pretty_print(file=file)
    elif self.data == 'return':
      print('return', end=' ', file=file)
      self.children['VAR'].pretty_print(file=file)
      print(';', file=file)

  @classproperty
  @classmethod
  def tokens(cls):
    return ['seq', 'return']

# I -> V <- F # assign
class InstNode(Tree):
  def __init__(self, *args, **kwargs):
    super(InstNode, self).__init__(*args, **kwargs)
    # InstNode has only one production rule
    self.data = 'assign'
    self.children = {
      'VAR': VarNode(parent=self, assignment=True),
      'FUNC': FuncNode(parent=self)
    }

  def production_space(self, used_vars=set()):
    for key in ['FUNC', 'VAR']:
      node, space, used_vars = self.children[key].production_space(used_vars=used_vars)
      if len(space) > 0:
        return node, space, used_vars
    return self, [], used_vars

  def production(self, rule=None):
    raise ValueError('InstNode should not have any hole.')
  
  def interprete(self, mem={}):
    if self.data == 'assign':
      value = self.children['FUNC'].interprete(mem)
      mem[self.children['VAR'].data] = value
      return mem

  def pretty_print(self, file=None):
    self.children['VAR'].pretty_print(file=file)
    print(' <- ', end='', file=file)
    self.children['FUNC'].pretty_print(file=file)

  @classproperty
  @classmethod
  def tokens(cls):
    return ['assign']

# V -> v0 | v1        # inputs
#    | v2 | ... | v19 # variables
class VarNode(Tree):
  VARIABLE_RANGE = 20
  VAR_SPACE = ['v{}'.format(i) for i in range(20)]
  INPUT_VARS = ['v0', 'v1']

  def __init__(self, assignment=False, *args, **kwargs):
    super(VarNode, self).__init__(*args, **kwargs)
    self.assignment = assignment

  def production_space(self, used_vars=set()):
    if self.data == 'hole' and self.assignment:
      space = [e for e in self.VAR_SPACE if e not in used_vars]
      if len(space) > 0:
        space = list(used_vars - set(self.INPUT_VARS)) + space[:1]
      else:
        space = list(used_vars - set(self.INPUT_VARS))
      return self, space, used_vars
    elif self.data == 'hole' and not self.assignment:
      self.used_vars = list(used_vars)
      return self, list(used_vars), used_vars
    elif self.assignment:
      used_vars.add(self.data)
      return self, [], used_vars
    else:
      return self, [], used_vars
    
  def production(self, rule=None):
    if self.assignment and rule in self.VAR_SPACE:
      self.data = rule
    elif not self.assignment and rule in self.used_vars:
      self.data = rule

  def interprete(self, mem):
    return mem[self.data]
    
  def pretty_print(self, file=None):
    print(self.data, end='', file=file)

  @classproperty
  @classmethod
  def tokens(cls):
    return cls.VAR_SPACE

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
  AUOP_FUNC = ['map']
  BUOP_FUNC = ['filter', 'count']
  ABOP_FUNC = ['zipwith', 'scanl']
  ONE_VAR_FUNC = ['head', 'last', 'minimum', 'maximum', 'reverse', 'sort', 'sum']
  TWO_VAR_FUNC = ['take', 'drop', 'access']

  def production_space(self, used_vars=set()):
    if self.data == 'hole':
      return self, self.AUOP_FUNC + self.BUOP_FUNC + self.ABOP_FUNC + self.ONE_VAR_FUNC + self.TWO_VAR_FUNC, used_vars
    keys = []
    if self.data in self.AUOP_FUNC:
      keys = ['AUOP', 'VAR']
    if self.data in self.BUOP_FUNC:
      keys = ['BUOP', 'VAR']
    if self.data in self.ABOP_FUNC:
      keys = ['ABOP', 'VAR1', 'VAR2']
    if self.data in self.ONE_VAR_FUNC:
      keys = ['VAR']
    if self.data in self.TWO_VAR_FUNC:
      keys = ['VAR1', 'VAR2']
    for key in keys:
      node, space, used_vars = self.children[key].production_space(used_vars=used_vars)
      if len(space) > 0:
        return node, space, used_vars
    return self, [], used_vars

  def production(self, rule=None):
    if rule in self.AUOP_FUNC:
      self.data = rule
      self.children = {
        'AUOP': AUOPNode(parent=self),
        'VAR': VarNode(parent=self)
      }
    elif rule in self.BUOP_FUNC:
      self.data = rule
      self.children = {
        'BUOP': BUOPNode(parent=self),
        'VAR': VarNode(parent=self)
      }
    elif rule in self.ABOP_FUNC:
      self.data = rule
      self.children = {
        'ABOP': ABOPNode(parent=self),
        'VAR1': VarNode(parent=self),
        'VAR2': VarNode(parent=self)
      }
    elif rule in self.ONE_VAR_FUNC:
      self.data = rule
      self.children = {
        'VAR': VarNode(parent=self)
      }
    elif rule in self.TWO_VAR_FUNC:
      self.data = rule
      self.children = {
        'VAR1': VarNode(parent=self),
        'VAR2': VarNode(parent=self)
      }

  def interprete(self, mem={}):
    if self.data == 'map':
      f = self.children['AUOP'].interprete()
      xs = self.children['VAR'].interprete(mem)
      if not isinstance(xs, IntList):
        raise UndefinedSemantics('type(xs): {}'.format(type(xs)))
      return IntList([f(x) for x in xs])

    if self.data == 'filter':
      f = self.children['BUOP'].interprete()
      xs = self.children['VAR'].interprete(mem)
      if not isinstance(xs, IntList):
        raise UndefinedSemantics('type(xs): {}'.format(type(xs)))
      return IntList([x for x in xs if f(x)])

    if self.data == 'count':
      f = self.children['BUOP'].interprete()
      xs = self.children['VAR'].interprete(mem)
      if not isinstance(xs, IntList):
        raise UndefinedSemantics('type(xs): {}'.format(type(xs)))
      return Integer(len([x for x in xs if f(x)]))

    if self.data == 'zipwith':
      f = self.children['ABOP'].interprete()
      xs = self.children['VAR1'].interprete(mem)
      if not isinstance(xs, IntList):
        raise UndefinedSemantics('type(xs): {}'.format(type(xs)))
      ys = self.children['VAR2'].interprete(mem)
      if not isinstance(ys, IntList):
        raise UndefinedSemantics('type(ys): {}'.format(type(ys)))
      return IntList([f(x, y) for x, y in zip(xs, ys)])

    if self.data == 'scanl':
      raise UndefinedSemantics()

    if self.data == 'head':
      xs = self.children['VAR'].interprete(mem)
      if not isinstance(xs, IntList):
        raise UndefinedSemantics('type(xs): {}'.format(type(xs)))
      if len(xs) == 0:
        raise UndefinedSemantics('len(xs) == 0')
      return xs[0]

    if self.data == 'last':
      xs = self.children['VAR'].interprete(mem)
      if not isinstance(xs, IntList):
        raise UndefinedSemantics('type(xs): {}'.format(type(xs)))
      if len(xs) == 0:
        raise UndefinedSemantics('len(xs) == 0')
      return xs[-1]

    if self.data == 'minimum':
      xs = self.children['VAR'].interprete(mem)
      if not isinstance(xs, IntList):
        raise UndefinedSemantics('type(xs): {}'.format(type(xs)))
      if len(xs) == 0:
        raise UndefinedSemantics('len(xs) == 0')
      return min(xs)

    if self.data == 'maximum':
      xs = self.children['VAR'].interprete(mem)
      if not isinstance(xs, IntList):
        raise UndefinedSemantics('type(xs): {}'.format(type(xs)))
      if len(xs) == 0:
        raise UndefinedSemantics('len(xs) == 0')
      return max(xs)

    if self.data == 'reverse':
      xs = self.children['VAR'].interprete(mem)
      if not isinstance(xs, IntList):
        raise UndefinedSemantics('type(xs): {}'.format(type(xs)))
      return reversed(xs)

    if self.data == 'sort':
      xs = self.children['VAR'].interprete(mem)
      if not isinstance(xs, IntList):
        raise UndefinedSemantics('type(xs): {}'.format(type(xs)))
      return IntList(sorted(xs))

    if self.data == 'sum':
      xs = self.children['VAR'].interprete(mem)
      if not isinstance(xs, IntList):
        raise UndefinedSemantics('type(xs): {}'.format(type(xs)))
      return sum(xs, ZERO)

    if self.data == 'take':
      n = self.children['VAR1'].interprete(mem)
      if not isinstance(n, Integer):
        raise UndefinedSemantics('type(n): {}'.format(type(n)))
      xs = self.children['VAR2'].interprete(mem)
      if not isinstance(xs, IntList):
        raise UndefinedSemantics('type(xs): {}'.format(type(xs)))
      return IntList(xs[:n])

    if self.data == 'drop':
      n = self.children['VAR1'].interprete(mem)
      if not isinstance(n, Integer):
        raise UndefinedSemantics('type(n): {}'.format(type(n)))
      xs = self.children['VAR2'].interprete(mem)
      if not isinstance(xs, IntList):
        raise UndefinedSemantics('type(xs): {}'.format(type(xs)))
      return IntList(xs[n:])
      
    if self.data == 'access':
      n = self.children['VAR1'].interprete(mem)
      if not isinstance(n, Integer):
        raise UndefinedSemantics('type(n): {}'.format(type(n)))
      xs = self.children['VAR2'].interprete(mem)
      if not isinstance(xs, IntList):
        raise UndefinedSemantics('type(xs): {}'.format(type(xs)))
      if len(xs) <= n.get_value():
        raise UndefinedSemantics('len(xs) <= n: {} <= {}'.format(len(xs), n))
      if len(xs) < -(n.get_value()):
        raise UndefinedSemantics('len(xs) < n: {} < {}'.format(len(xs), -n))
      return xs[n]

  def pretty_print(self, file=None):
    if self.data == 'hole':
      print('(HOLE)', end='', file=file)
    elif self.data in self.AUOP_FUNC:
      print(self.data.upper(), end=' ', file=file)
      self.children['AUOP'].pretty_print(file=file)
      print(end=' ', file=file)
      self.children['VAR'].pretty_print(file=file)
    elif self.data in self.BUOP_FUNC:
      print(self.data.upper(), end=' ', file=file)
      self.children['BUOP'].pretty_print(file=file)
      print(end=' ', file=file)
      self.children['VAR'].pretty_print(file=file)
    elif self.data in self.ABOP_FUNC:
      print(self.data.upper(), end=' ', file=file)
      self.children['ABOP'].pretty_print(file=file)
      print(end=' ', file=file)
      self.children['VAR1'].pretty_print(file=file)
      print(end=' ', file=file)
      self.children['VAR2'].pretty_print(file=file)
    elif self.data in self.ONE_VAR_FUNC:
      print(self.data.upper(), end=' ', file=file)
      self.children['VAR'].pretty_print(file=file)
    elif self.data in self.TWO_VAR_FUNC:
      print(self.data.upper(), end=' ', file=file)
      self.children['VAR1'].pretty_print(file=file)
      print(end=' ', file=file)
      self.children['VAR2'].pretty_print(file=file)

  @classproperty
  @classmethod
  def tokens(cls):
    return cls.AUOP_FUNC + cls.BUOP_FUNC + cls.ABOP_FUNC + cls.ONE_VAR_FUNC + cls.TWO_VAR_FUNC

# AUOP -> +1 | -1 | *2 | /2 | *(-1) | **2 | *3 | /3 | *4 | /4
class AUOPNode(Tree):
  AUOP_SPACE = ['+1', '-1', '*2', '/2', '*(-1)', '**2', '*3', '/3', '*4', '/4']

  def production_space(self, used_vars=set()):
    if self.data == 'hole':
      return self, self.AUOP_SPACE, used_vars
    else:
      return self, [], used_vars

  def production(self, rule=None):
    if rule in self.AUOP_SPACE:
      self.data = rule

  def interprete(self):
    if self.data == '+1':
      return lambda x: x + ONE
    if self.data == '-1':
      return lambda x: x - ONE
    if self.data == '*2':
      return lambda x: x * TWO
    if self.data == '/2':
      return lambda x: x // TWO
    if self.data == '*(-1)':
      return lambda x: -x
    if self.data == '**2':
      return lambda x: x * x
    if self.data == '*3':
      return lambda x: x * THREE
    if self.data == '/3':
      return lambda x: x / THREE
    if self.data == '*4':
      return lambda x: x * FOUR
    if self.data == '/4':
      return lambda x: x / FOUR
  
  def pretty_print(self, file=None):
    print('({})'.format(self.data), end='', file=file)

  @classproperty
  @classmethod
  def tokens(cls):
    return cls.AUOP_SPACE


# BUOP -> >0 | <0 | %2==0 | %2==1
class BUOPNode(Tree):
  BUOP_SPACE = ['>0', '<0', '%2==0', '%2==1']

  def production_space(self, used_vars=set()):
    if self.data == 'hole':
      return self, self.BUOP_SPACE, used_vars
    else:
      return self, [], used_vars
    
  def production(self, rule=None):
    if rule in self.BUOP_SPACE:
      self.data = rule

  def interprete(self):
    if self.data == '>0':
      return lambda x: x > ZERO
    if self.data == '<0':
      return lambda x: x < ZERO
    if self.data == '%2==0':
      return lambda x: (x % TWO) == ZERO
    if self.data == '%2==1':
      return lambda x: (x % TWO) == ONE
    
  def pretty_print(self, file=None):
    print('({})'.format(self.data), end='', file=file)

  @classproperty
  @classmethod
  def tokens(cls):
    return cls.BUOP_SPACE

# ABOP -> + | * | MIN | MAX
class ABOPNode(Tree):
  ABOP_SPACE = ['+', '*', 'MIN', 'MAX']

  def production_space(self, used_vars=set()):
    if self.data == 'hole':
      return self, self.ABOP_SPACE, used_vars
    else:
      return self, [], used_vars

  def production(self, rule=None):
    if rule in self.ABOP_SPACE:
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

  def pretty_print(self, file=None):
    print('({})'.format(self.data), end='', file=file)

  @classproperty
  @classmethod
  def tokens(cls):
    return cls.ABOP_SPACE
