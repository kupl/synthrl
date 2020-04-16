from itertools import cycle

from synthrl.language.dsl import Tree
from synthrl.language.dsl import UndefinedSemantics
from synthrl.utils.decoratorutils import classproperty
from synthrl.value import Integer
from synthrl.value import IntList
from synthrl.value import NoneType
from synthrl.value.integer import ZERO, ONE, TWO, THREE, FOUR

# wrapper class for program
# L -> P  # root
class ListLanguage(Tree):
  def __init__(self, input_types=None, output_type=None, *args, **kwargs):
    self.data = 'root'
    self.children = {'PGM': ProgramNode(parent=self)}
    self.parent = None
    input_types = list(input_types)
    if len(input_types) > len(VarNode.INPUT_VARS):
      raise ValueError('Expecting at most {} inputs, but {} inputs are given.'.format(len(VarNode.INPUT_VARS), len(input_types)))
    if len(input_types) >= 1:
      if not (input_types[0] == list or input_types[0] == IntList):
        raise ValueError('Argument 0 must be list of integers, but {} is given.'.format(input_types[0]))
      input_types[0] = IntList
    for i in range(1, len(VarNode.INPUT_VARS)):
      try:
        t = input_types[i]
        if t is None or t == NoneType:
          input_types[i] = NoneType
        elif t == int or t == Integer:
          input_types[i] = Integer
        elif t == list or t == IntList:
          input_types[i] = IntList
        else:
          raise ValueError('Unsupported Value {} is given.'.format(t))
      except IndexError:
        input_types.append(NoneType)
    self.input_types = input_types
    if output_type == int or output_type == Integer:
      self.output_type = Integer
    elif output_type == list or output_type == IntList:
      self.output_type = IntList
    else:
      raise ValueError('Return type must be integer or list, but {} is given.'.format(output_type))

  def production_space(self):
    used_vars = {v: t for v, t in zip(VarNode.INPUT_VARS, self.input_types)}
    node, space, _ = self.children['PGM'].production_space(used_vars=used_vars)
    return node, space
  
  def production(self, rule=None):
    raise ValueError('ListLanguage should not have any hole.')

  def interprete(self, inputs=[]):
    mem = {}
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

  def copy(self):
    node = self.__class__(input_types=self.input_types, output_type=self.output_type)
    children = {}
    for key, child in self.children.items():
      child = child.copy()
      child.parent = node
      children[key] = child
    node.children = children
    return node

# P -> I; P       # seq
#    | return V;  # return
class ProgramNode(Tree):
  def production_space(self, used_vars={}):
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
      root = self.parent
      while root.parent:
        root = root.parent
      self.data = 'return'
      self.children = {
        'VAR': VarNode(parent=self, typ=root.output_type)
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

  def production_space(self, used_vars={}):
    node, space, used_vars = self.children['FUNC'].production_space(used_vars=used_vars)
    if len(space) > 0:
      return node, space, used_vars
    self.children['VAR'].type = self.children['FUNC'].return_type
    node, space, used_vars = self.children['VAR'].production_space(used_vars=used_vars)
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

# V -> a0 | a1        # inputs
#    | v1 | ... | v9  # variables
class VarNode(Tree):
  INPUT_VARS = ['a{}'.format(i) for i in range(2)]
  VAR_SPACE = ['v{}'.format(i) for i in range(10)]

  def __init__(self, assignment=False, typ=None, *args, **kwargs):
    super(VarNode, self).__init__(*args, **kwargs)
    self.assignment = assignment
    self.type = typ

  def production_space(self, used_vars={}):
    if self.data == 'hole' and self.assignment:
      self.possible = [k for k in used_vars.keys() if k not in self.INPUT_VARS]
      if len(self.possible) < len(self.VAR_SPACE):
        self.possible.append(self.VAR_SPACE[len(self.possible)])
      return self, self.possible, used_vars
    elif self.data == 'hole' and not self.assignment:
      self.possible = [k for k, t in used_vars.items() if t == self.type]
      print('used_vars', used_vars)
      print('possible', self.possible)
      return self, self.possible, used_vars
    elif self.assignment:
      used_vars[self.data] = self.type
      return self, [], used_vars
    else:
      return self, [], used_vars
    
  def production(self, rule=None):
    if rule in self.possible:
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
#    | SCANL1 ABOP V    # scanl1
#    | ZIPWITH ABOP V V # zipwith
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
  AUOP_FUNC_RETL = ['map']
  BUOP_FUNC_RETL = ['filter']
  BUOP_FUNC_RETI = ['count']
  ABOP1_FUNC_RETL = ['scanl1']
  ABOP2_FUNC_RETL = ['zipwith']
  ONE_VAR_FUNC_RETL = ['reverse', 'sort']
  ONE_VAR_FUNC_RETI = ['head', 'last', 'minimum', 'maximum', 'sum']
  TWO_VAR_FUNC_RETL = ['take', 'drop']
  TWO_VAR_FUNC_RETI = ['access']

  def production_space(self, used_vars=set()):
    if self.data == 'hole':
      return self, self.tokens, used_vars
    keys = []
    if self.data in self.AUOP_FUNC_RETL:
      keys = ['AUOP', 'VAR']
    if self.data in self.BUOP_FUNC_RETL + self.BUOP_FUNC_RETI:
      keys = ['BUOP', 'VAR']
    if self.data in self.ABOP1_FUNC_RETL:
      keys = ['ABOP', 'VAR']
    if self.data in self.ABOP2_FUNC_RETL:
      keys = ['ABOP', 'VAR1', 'VAR2']
    if self.data in self.ONE_VAR_FUNC_RETL + self.ONE_VAR_FUNC_RETI:
      keys = ['VAR']
    if self.data in self.TWO_VAR_FUNC_RETL + self.TWO_VAR_FUNC_RETI:
      keys = ['VAR1', 'VAR2']
    for key in keys:
      node, space, used_vars = self.children[key].production_space(used_vars=used_vars)
      if len(space) > 0:
        return node, space, used_vars
    return self, [], used_vars

  def production(self, rule=None):
    if rule in self.AUOP_FUNC_RETL:
      self.data = rule
      self.children = {
        'AUOP': AUOPNode(parent=self),
        'VAR': VarNode(parent=self, typ=IntList)
      }
    elif rule in self.BUOP_FUNC_RETL + self.BUOP_FUNC_RETI:
      self.data = rule
      self.children = {
        'BUOP': BUOPNode(parent=self),
        'VAR': VarNode(parent=self, typ=IntList)
      }
    elif rule in self.ABOP1_FUNC_RETL:
      self.data = rule
      self.children = {
        'ABOP': ABOPNode(parent=self),
        'VAR': VarNode(parent=self, typ=IntList)
      }
    elif rule in self.ABOP2_FUNC_RETL:
      self.data = rule
      self.children = {
        'ABOP': ABOPNode(parent=self),
        'VAR1': VarNode(parent=self, typ=IntList),
        'VAR2': VarNode(parent=self, typ=IntList)
      }
    elif rule in self.ONE_VAR_FUNC_RETL + self.ONE_VAR_FUNC_RETI:
      self.data = rule
      self.children = {
        'VAR': VarNode(parent=self, typ=IntList)
      }
    elif rule in self.TWO_VAR_FUNC_RETL + self.TWO_VAR_FUNC_RETI:
      self.data = rule
      self.children = {
        'VAR1': VarNode(parent=self, typ=Integer),
        'VAR2': VarNode(parent=self, typ=IntList)
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

    if self.data == 'scanl1':
      f = self.children['ABOP'].interprete()
      xs = self.children['VAR'].interprete(mem)
      if not isinstance(xs, IntList):
        raise UndefinedSemantics('type(xs): {}'.format(type(xs)))
      if len(xs) == 0:
        return IntList()
      running_value = xs[0]
      ys = IntList([running_value])
      for x in xs[1:]:
        running_value = f(running_value, x)
        ys.append(running_value)
      return ys

    if self.data == 'zipwith':
      f = self.children['ABOP'].interprete()
      xs = self.children['VAR1'].interprete(mem)
      if not isinstance(xs, IntList):
        raise UndefinedSemantics('type(xs): {}'.format(type(xs)))
      ys = self.children['VAR2'].interprete(mem)
      if not isinstance(ys, IntList):
        raise UndefinedSemantics('type(ys): {}'.format(type(ys)))
      return IntList([f(x, y) for x, y in zip(xs, ys)])

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
    elif self.data in self.AUOP_FUNC_RETL:
      print(self.data.upper(), end=' ', file=file)
      self.children['AUOP'].pretty_print(file=file)
      print(end=' ', file=file)
      self.children['VAR'].pretty_print(file=file)
    elif self.data in self.BUOP_FUNC_RETL + self.BUOP_FUNC_RETI:
      print(self.data.upper(), end=' ', file=file)
      self.children['BUOP'].pretty_print(file=file)
      print(end=' ', file=file)
      self.children['VAR'].pretty_print(file=file)
    elif self.data in self.ABOP1_FUNC_RETL:
      print(self.data.upper(), end=' ', file=file)
      self.children['ABOP'].pretty_print(file=file)
      print(end=' ', file=file)
      self.children['VAR'].pretty_print(file=file)
    elif self.data in self.ABOP2_FUNC_RETL:
      print(self.data.upper(), end=' ', file=file)
      self.children['ABOP'].pretty_print(file=file)
      print(end=' ', file=file)
      self.children['VAR1'].pretty_print(file=file)
      print(end=' ', file=file)
      self.children['VAR2'].pretty_print(file=file)
    elif self.data in self.ONE_VAR_FUNC_RETL + self.ONE_VAR_FUNC_RETI:
      print(self.data.upper(), end=' ', file=file)
      self.children['VAR'].pretty_print(file=file)
    elif self.data in self.TWO_VAR_FUNC_RETL + self.TWO_VAR_FUNC_RETI:
      print(self.data.upper(), end=' ', file=file)
      self.children['VAR1'].pretty_print(file=file)
      print(end=' ', file=file)
      self.children['VAR2'].pretty_print(file=file)

  @classproperty
  @classmethod
  def tokens(cls):
    return cls.AUOP_FUNC_RETL + cls.BUOP_FUNC_RETL + cls.BUOP_FUNC_RETI + cls.ABOP1_FUNC_RETL + cls.ABOP2_FUNC_RETL + cls.ONE_VAR_FUNC_RETL + cls.ONE_VAR_FUNC_RETI + cls.TWO_VAR_FUNC_RETL + cls.TWO_VAR_FUNC_RETI

  @property
  def return_type(self):
    if self.data in self.AUOP_FUNC_RETL + self.BUOP_FUNC_RETL + self.ABOP1_FUNC_RETL + self.ABOP2_FUNC_RETL + self.ONE_VAR_FUNC_RETL + self.TWO_VAR_FUNC_RETL:
      return IntList
    elif self.data in self.BUOP_FUNC_RETI + self.ONE_VAR_FUNC_RETI + self.TWO_VAR_FUNC_RETI:
      return Integer

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
