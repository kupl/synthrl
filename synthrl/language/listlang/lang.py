# grammar for ListLang
# L -> x_1 <- I;
#      x_2 <- I;
#      ...;
#      x_T <- I;              # root: T sequences
# I -> MAP (AUOP) V           # map
#    | FILTER (BUOP) V        # filter
#    | COUNT (BUOP) V         # count
#    | SCANL1 (ABOP) V        # scanl1
#    | ZIPWITH (ABOP) V V     # zipwith
#    | HEAD V                 # head
#    | LAST V                 # last
#    | MINIMUM V              # minimum
#    | MAXIMUM V              # maximum
#    | REVERSE V              # reverse
#    | SORT V                 # sort
#    | SUM V                  # sum
#    | TAKE V V               # take
#    | DROP V V               # drop
#    | ACCESS V V             # access
#    | NOP                    # nop
# V -> a_1 | a_2 | ... | a_S  # inputs
#    | x_1 | x_2 | ... | x_T  # variables
# AUOP -> +1 | -1 | *2 | /2 | *(-1) | **2 | *3 | /3 | *4 | /4
# BUOP -> POS | NEG | EVEN | ODD
# ABOP -> + | * | MIN | MAX

from synthrl.language.abstract import Node
from synthrl.language.abstract import Tree
from synthrl.language.abstract import UndefinedSemantics
from synthrl.language.abstract import WrongProductionException
from synthrl.utils.decoratorutils import classproperty
from synthrl.utils.exceptionutils import UnexpectedException
from synthrl.value import Integer
from synthrl.value import IntList
from synthrl.value.integer import FOUR
from synthrl.value.integer import ONE
from synthrl.value.integer import THREE
from synthrl.value.integer import TWO
from synthrl.value.integer import ZERO
from synthrl.value.nonetype import NONE

# maximum number of inputs
S = 2
# maximum length of sequences
T = 5
# symbol L
class ListLang(Tree):
  
  __slots__ = [
    'input_types',
    'instructions',
    'output_type',
  ]

  def __init__(self, input_types, output_type=None):\
    # input_types: A tuple or list of types of inputs.
    #              The size must be 1 or 2 and the first element must be IntList(or list).
    # output_type: The type of the program output.
    #              None represents "any" type, which is default.

    super(ListLang, self).__init__()

    # check input types
    self.input_types = []
    for ty in input_types:
      if ty == int or ty == Integer:
        self.input_types.append(Integer)
      elif ty == list or ty == IntList:
        self.input_types.append(IntList)
      else:
        raise ValueError('Invalid type "{}" is given.'.format(ty.__name__))
    if len(self.input_types) == 0:
      raise ValueError('No input type is given')
    if len(self.input_types) > S:
      raise ValueError('Expecting at most {} inputs, but {} inputs are given.'.format(S, len(self.input_types)))
    if self.input_types[0] != IntList:
      raise ValueError('First argument must be IntList, but {} is given.'.format(self.input_types[0].__name__))
    
    # check output type
    if output_type == int or output_type == Integer:
      self.output_type = Integer
    elif output_type == list or output_type == IntList:
      self.output_type = IntList
    elif output_type is None:
      self.output_type = None
    else:
      raise ValueError('Invalid output type "{}" is given.'.format(output_type.__name__))
    
    # instructions
    self.instructions = [InstNode(parent=self) for _ in range(T)]

  @property
  def production_space(self):

    # variables track types of variables
    last_type = self.input_types[-1]
    var_types = {'a_{}'.format(i + 1): ty for i, ty in enumerate(self.input_types)}

    # for each instruction
    for i, inst in enumerate(self.instructions):

      # try find a hole to fill
      self.hole, space = inst.production_space(loc=i, last_type=last_type, return_type=self.output_type, var_types=var_types)

      # if hole found
      if len(space) > 0:
        break

      # if program ends
      if inst.data == 'nop':
        break

      # update variables and go to next instruction
      last_type = inst.return_type
      var_types['v_{}'.format(i + 1)] = inst.return_type

    return space

  def interprete(self, inputs):
    # inputs: inputs to execute the program

    # convert python values to synthrl.value types
    if len(inputs) != len(self.input_types):
      raise UndefinedSemantics('Expecting {} inputs, but {} inputs are given.'.format(len(self.input_types), len(inputs)))
    inputs = [ty(i) for i, ty in zip(inputs, self.input_types)]

    # initialize a dictionay that contains variables and their values with inputs
    mem = {'a_{}'.format(i + 1): v for i, v in enumerate(inputs)}

    # run each instructions
    for i, inst in enumerate(self.instructions):
      x_i = inst.interprete(mem)
      mem['x_{}'.format(i + 1)] = x_i

    # return the result of the final instruction
    return x_i

  def pretty_print(self, file=None):
    # file: TextIOWrapper to use as write stream. By default, use stdout.

    # print inputs
    for i, ty in enumerate(self.input_types):
      print('a_{} <- {};'.format(i + 1, 'int' if ty == Integer else '[int]'), file=file)

    # print instructions
    for i, inst in enumerate(self.instructions):
      print('x_{} <- '.format(i + 1), end='', file=file)
      inst.pretty_print(file=file)
      print(';', file=file)

  @classproperty
  @classmethod
  def tokens(cls):
    return InstNode.tokens + VarNode.tokens + AUOPNode.tokens + BUOPNode.tokens + ABOPNode.tokens

# symbol I
class InstNode(Node):

  # dictionary has properties of tokens
  # TOKENS = {token: [options, # of parameters, return type]}
  TOKENS = {
    'map':     ['AUOP'  , 1, 'LIST'],
    'filter':  ['BUOP'  , 1, 'LIST'],
    'count':   ['BUOP'  , 1, 'INT' ],
    'scanl1':  ['ABOP'  , 1, 'LIST'],
    'zipwith': ['ABOP'  , 2, 'LIST'],
    'head':    ['NOOPT' , 1, 'INT' ],
    'last':    ['NOOPT' , 1, 'INT' ],
    'minimum': ['NOOPT' , 1, 'INT' ],
    'maximum': ['NOOPT' , 1, 'INT' ],
    'reverse': ['NOOPT' , 1, 'LIST'],
    'sort':    ['NOOPT' , 1, 'LIST'],
    'sum':     ['NOOPT' , 1, 'INT' ],
    'take':    ['REQINT', 2, 'LIST'],
    'drop':    ['REQINT', 2, 'LIST'],
    'access':  ['REQINT', 2, 'INT' ],
    'nop':     ['END'   , 1, 'LIST'],
  }
  
  def production_space(self, loc, var_types, last_type, return_type=None):
    # loc        : Location of instruction. Must be smaller than T.
    # var_types  : A dictionary contains the type information of previously assigned variable.
    # last_type  : The return type of very before instruction.
    # return_type: The desired return type of the whole program.
    
    # if the node should be filled
    if self.data == 'HOLE':

      # options that can be used
      options = ['AUOP', 'BUOP', 'ABOP', 'NOOPT']
      if return_type and last_type == return_type:
        # possibly end of program
        options.append('END')
      if Integer in [e for _, e in var_types.items()]:
        # can use functions that require integer
        options.append('REQINT')
    
      # possible return type
      types = []
      if loc == T and return_type:
        # must be ended
        types.append('LIST' if return_type == IntList else 'Int')
      else:
        types.extend(['LIST', 'INT'])
    
      # returns the token that satisfy conditions
      return self, [token for token, prop in self.TOKENS if prop[0] in options and prop[2] in types]
    
    # if program ended
    elif self.data == 'nop':

      # no hole
      return self, []

    # otherwise
    else:

      # find keys and their order for searching
      keys = []
      option, n_vars, _ = self.TOKENS[self.data]
      # map
      if option == 'AUOP' and n_vars == 1:
        keys.extend(['AUOP', 'VAR'])
      # filter, count
      elif option == 'BUOP' and n_vars == 1:
        keys.extend(['BUOP', 'VAR'])
      # scanl1
      elif option == 'ABOP' and n_vars == 1:
        keys.extend(['ABOP', 'VAR'])
      # zipwith
      elif option == 'ABOP' and n_vars == 2:
        keys.extend(['ABOP', 'VAR1', 'VAR2'])
      # head, last, minimum, maximum, reverse, sort, sum
      elif option == 'NOOPT' and n_vars == 1:
        keys.append('VAR')
      # take, drop, access
      elif option == 'REQINT' and n_vars == 2:
        keys.extend(['VAR1', 'VAR2'])
      # should not reach here
      else:
        raise UnexpectedException('Unexpected values in "InstNode.production_space". {{option: {}, n_vars: {}}}'.format(option, n_vars))
      
      # search the hole and production space
      for key in keys:
        node, space = self.children[key].production_space(var_types)
        if len(space) > 0:
          return node, space
      # if all nodes are filed
      return self, []

  def production(self, rule):
    # rule: A rule to apply to this node.

    # check the given rule is valid and update
    if rule not in self.TOKENS.keys():
      raise WrongProductionException('Invalid production rule "{}" is given.'.format(rule))
    self.data = rule
    
    # create children
    option, n_vars, _ = self.TOKENS[self.data]
    # map
    if option == 'AUOP' and n_vars == 1:
      self.children = {
        'AUOP': AUOPNode(parent=self),
        'VAR': VarNode(parent=self, type=IntList)
      }
    # filter, count
    elif option == 'BUOP' and n_vars == 1:
      self.children = {
        'BUOP': BUOPNode(parent=self),
        'VAR': VarNode(parent=self, type=IntList)
      }
    # scanl1
    elif option == 'ABOP' and n_vars == 1:
      self.children = {
        'ABOP': ABOPNode(parent=self),
        'VAR': VarNode(parent=self, type=IntList)
      }
    # zipwith
    elif option == 'ABOP' and n_vars == 2:
      self.children = {
        'ABOP': ABOPNode(parent=self),
        'VAR1': VarNode(parent=self, type=IntList),
        'VAR2': VarNode(parent=self, type=IntList)
      }
    # head, last, minimum, maximum, reverse, sort, sum
    elif option == 'NOOPT' and n_vars == 1:
      self.children = {
        'VAR': VarNode(parent=self, type=IntList)
      }
    # take, drop, access
    elif option == 'REQINT' and n_vars == 2:
      self.children = {
        'VAR1': VarNode(parent=self, type=Integer),
        'VAR2': VarNode(parent=self, type=IntList)
      }
    # should not reach here
    else:
      raise UnexpectedException('Unexpected values in "InstNode.production_space". {{option: {}, n_vars: {}}}'.format(option, n_vars))

  def interprete(self, mem):
    # mem:  A dictionary that contains assigned variables and their values.
    
    # map
    if self.data == 'map':
      f = self.children['AUOP'].interprete()
      xs = self.children['VAR'].interprete(mem)
      return IntList(map(f, xs))
    
    # filter
    elif self.data == 'filter':
      f = self.children['AUOP'].interprete()
      xs = self.children['VAR'].interprete(mem)
      return IntList(filter(f, xs))
    
    # count
    elif self.data == 'count':
      f = self.children['AUOP'].interprete()
      xs = self.children['VAR'].interprete(mem)
      return Integer(len(list(filter(f, xs))))
    
    # scanl1
    elif self.data == 'scanl1':
      f = self.children['ABOP'].interprete()
      xs = self.children['VAR'].interprete(mem)
      if len(xs) == 0:
        return IntList()
      running_value = xs[0]
      ys = IntList([running_value])
      for x in xs[1:]:
        running_value = f(running_value, x)
        ys.append(running_value)
      return ys
    
    # zipwith
    elif self.data == 'zipwith':
      f = self.children['ABOP'].interprete()
      xs = self.children['VAR1'].interprete(mem)
      ys = self.children['VAR2'].interprete(mem)
      return IntList(map(lambda x: f(x[0], f[1]), zip(xs, ys)))
    
    # head
    elif self.data == 'head':
      xs = self.children['VAR'].interprete(mem)
      if len(xs) == 0:
        raise UndefinedSemantics('len(xs) == 0')
      return xs[0]
    
    # last
    elif self.data == 'last':
      xs = self.children['VAR'].interprete(mem)
      if len(xs) == 0:
        raise UndefinedSemantics('len(xs) == 0')
      return xs[-1]
    
    # minimum
    elif self.data == 'minimum':
      xs = self.children['VAR'].interprete(mem)
      if len(xs) == 0:
        raise UndefinedSemantics('len(xs) == 0')
      return min(xs)
    
    # maximum
    elif self.data == 'maximum':
      xs = self.children['VAR'].interprete(mem)
      if len(xs) == 0:
        raise UndefinedSemantics('len(xs) == 0')
      return max(xs)
    
    # reverse
    elif self.data == 'reverse':
      xs = self.children['VAR'].interprete(mem)
      return reversed(xs)
    
    # sort
    elif self.data == 'sort':
      xs = self.children['VAR'].interprete(mem)
      return IntList(sorted(xs))
    
    # sum
    elif self.data == 'sum':
      xs = self.children['VAR'].interprete(mem)
      return sum(xs, ZERO)
    
    # take
    elif self.data == 'take':
      n = self.children['VAR1'].interprete(mem)
      xs = self.children['VAR2'].interprete(mem)
      return xs[:n]
    
    # drop
    elif self.data == 'drop':
      n = self.children['VAR1'].interprete(mem)
      xs = self.children['VAR2'].interprete(mem)
      return xs[n:]
    
    # access
    elif self.data == 'access':
      n = self.children['VAR1'].interprete(mem)
      xs = self.children['VAR2'].interprete(mem)
      if len(xs) <= n.get_value():
        raise UndefinedSemantics('len(xs) <= n: {} <= {}'.format(len(xs), n))
      if len(xs) < -(n.get_value()):
        raise UndefinedSemantics('len(xs) < n: {} < {}'.format(len(xs), -n))
      return xs[n]
    
    # nop
    elif self.data == 'nop':
      return NONE
    
    # should not reach here
    else:
      raise UnexpectedException('Unexpected value in "InstNode.interprete". {{self.data: {}}}'.format(self.data))

  def pretty_print(self, file=None):
    # file: TextIOWrapper to use as write stream. By default, use stdout.
    
    # if this node is hole
    if self.data == 'HOLE':
      print('(HOLE)', end='', file=file)
    
    # otherwise
    else:

      # distinguish token
      option, n_vars, _ = self.TOKENS[self.data]
      # map
      if option == 'AUOP' and n_vars == 1:
        print('{} ('.format(self.data.upper()), end='', file=file)
        self.children['AUOP'].pretty_print(file=file)
        print(') ', end='', file=file)
        self.children['VAR'].pretty_print(file=file)
      # filter, count
      elif option == 'BUOP' and n_vars == 1:
        print('{} ('.format(self.data.upper()), end='', file=file)
        self.children['BUOP'].pretty_print(file=file)
        print(') ', end='', file=file)
        self.children['VAR'].pretty_print(file=file)
      # scanl1
      elif option == 'ABOP' and n_vars == 1:
        print('{} ('.format(self.data.upper()), end='', file=file)
        self.children['ABOP'].pretty_print(file=file)
        print(') ', end='', file=file)
        self.children['VAR'].pretty_print(file=file)
      # zipwith
      elif option == 'ABOP' and n_vars == 2:
        print('{} ('.format(self.data.upper()), end='', file=file)
        self.children['ABOP'].pretty_print(file=file)
        print(') ', end='', file=file)
        self.children['VAR1'].pretty_print(file=file)
        print(' ', end='', file=file)
        self.children['VAR2'].pretty_print(file=file)
      # head, last, minimum, maximum, reverse, sort, sum
      elif option == 'NOOPT' and n_vars == 1:
        print('{} '.format(self.data.upper()), end='', file=file)
        self.children['VAR'].pretty_print(file=file)
      # take, drop, access
      elif option == 'REQINT' and n_vars == 2:
        print('{} '.format(self.data.upper()), end='', file=file)
        self.children['VAR1'].pretty_print(file=file)
        print(' ', end='', file=file)
        self.children['VAR2'].pretty_print(file=file)
      # should not reach here
      else:
        raise UnexpectedException('Unexpected values in "InstNode.pretty_print". {{option: {}, n_vars: {}}}'.format(option, n_vars))

  @classproperty
  @classmethod
  def tokens(cls):
    return list(cls.TOKENS.keys())

  @property
  def return_type(self):
    # returns the return type of instruction
    return IntList if self.TOKENS[self.data][2] == 'LIST' else Integer

# symbol V
class VarNode(Node):

  __slots__ = ['type', 'vars']

  def __init__(self, type=IntList, *args, **kwargs):
    # type: The type of the variable. By default, IntList.

    super(VarNode, self).__init__(*args, **kwargs)

    # store the desiring type
    self.type = type
    self.vars = None

  def production_space(self, var_types):
    # var_types: A dictionary contains the type information of previously assigned variable.

    # if the node should be filled
    if self.data == 'HOLE':
      self.vars = [var for var, ty in var_types.items() if ty == self.type]
      return self, self.vars
    
    # if the node already filled
    else:
      return self, []

  def production(self, rule):
    # rule: A rule to apply to this node.

    # check the validity of rule
    if rule in self.vars:
      self.data = rule
    else:
      raise WrongProductionException('Invalid production rule "{}" is given.'.format(rule))

  def interprete(self, mem):
    # mem:  A dictionary that contains assigned variables and their values.

    # find and return data
    return mem[self.data]

  def pretty_print(self, file=None):
    # file: TextIOWrapper to use as write stream. By default, use stdout.
    
    # print a token
    print(self.data, end='', file=file)

  @classproperty
  @classmethod
  def tokens(cls):
    return ['a_{}'.format(i + 1) for i in range(S)] + ['x_{}'.format(i + 1) for i in range(T)]

# symbol AUOP
class AUOPNode(Node):

  # list of tokens
  TOKENS = ['+1', '-1', '*2', '/2', '*(-1)', '**2', '*3', '/3', '*4', '/4']
  
  def production_space(self, *args, **kwargs):
    
    # if the node should be filled
    if self.data == 'HOLE':
      return self, self.TOKENS

    # if the node is filled
    else:
      return self, []

  def production(self, rule):
    # rule: A rule to apply to this node.

    # check the validity of rule
    if rule in self.TOKENS:
      self.data = rule
    else:
      raise WrongProductionException('Invalid production rule "{}" is given.'.format(rule))

  def interprete(self):
    
    # +1
    if self.data == '+1':
      return lambda x: x + ONE

    # -1
    elif self.data == '-1':
      return lambda x: x - ONE

    # *2
    elif self.data == '*2':
      return lambda x: x * TWO
      
    # /2
    elif self.data == '/2':
      return lambda x: x // TWO
      
    # *(-1)
    elif self.data == '*(-1)':
      return lambda x: -x
      
    # **2
    elif self.data == '**2':
      return lambda x: x * x
      
    # *3
    elif self.data == '*3':
      return lambda x: x * THREE
      
    # /3
    elif self.data == '/3':
      return lambda x: x / THREE
      
    # *4
    elif self.data == '*4':
      return lambda x: x * FOUR
      
    # /4
    elif self.data == '/4':
      return lambda x: x / FOUR

    # should not reach here
    else:
      raise UnexpectedException('Unexpected value in "AUOPNode.interprete". {{self.data: {}}}'.format(self.data))

  def pretty_print(self, file=None):
    # file: TextIOWrapper to use as write stream. By default, use stdout.
    
    # print a token
    print(self.data, end='', file=file)

  @classproperty
  @classmethod
  def tokens(cls):
    return cls.TOKENS

# symbol BUOP
class BUOPNode(Node):
  
  # list of tokens
  TOKENS = ['pos', 'neg', 'even', 'odd']

  def production_space(self, *args, **kwargs):
    
    # if the node should be filled
    if self.data == 'HOLE':
      return self, self.TOKENS

    # if the node is filled
    else:
      return self, []

  def production(self, rule):
    # rule: A rule to apply to this node.

    # check the validity of rule
    if rule in self.TOKENS:
      self.data = rule
    else:
      raise WrongProductionException('Invalid production rule "{}" is given.'.format(rule))

  def interprete(self):
    
    # POS
    if self.data == 'pos':
      return lambda x: x > ZERO

    # NEG
    elif self.data == 'neg':
      return lambda x: x < ZERO

    # EVEN
    elif self.data == 'even':
      return lambda x: (x % TWO) == ZERO

    # ODD
    elif self.data == 'odd':
      return lambda x: (x % TWO) == ONE

    # should not reach here
    else:
      raise UnexpectedException('Unexpected value in "BUOPNode.interprete". {{self.data: {}}}'.format(self.data))

  def pretty_print(self, file=None):
    # file: TextIOWrapper to use as write stream. By default, use stdout.
    
    # print a token
    print(self.data.upper(), end='', file=file)

  @classproperty
  @classmethod
  def tokens(cls):
    return cls.TOKENS

# symbol ABOP
class ABOPNode(Node):

  # list of tokens
  TOKENS = ['+', '*', 'min', 'max']
  
  def production_space(self, *args, **kwargs):
    
    # if the node should be filled
    if self.data == 'HOLE':
      return self, self.TOKENS

    # if the node is filled
    else:
      return self, []

  def production(self, rule):
    # rule: A rule to apply to this node.

    # check the validity of rule
    if rule in self.TOKENS:
      self.data = rule
    else:
      raise WrongProductionException('Invalid production rule "{}" is given.'.format(rule))
    
  def interprete(self):

    # +
    if self.data == '+':
      return lambda x, y: x + y

    # *
    elif self.data == '*':
      return lambda x, y: x * y

    # MIN
    elif self.data == 'MIN':
      return lambda x, y: x if x < y else y

    # MAX
    elif self.data == 'MAX':
      return lambda x, y: x if x > y else y
    
    # should not reach here
    else:
      raise UnexpectedException('Unexpected value in "ABOPNode.interprete". {{self.data: {}}}'.format(self.data))

  def pretty_print(self, file=None):
    # file: TextIOWrapper to use as write stream. By default, use stdout.
    
    # print a token
    print(self.data.upper(), end='', file=file)

  @classproperty
  @classmethod
  def tokens(cls):
    return cls.TOKENS
