# grammar for ListLang
# L -> x_1 <- I;
#      x_2 <- I;
#      ...;
#      x_T <- I;              # root: T sequences
# I -> MAP AUOP V             # map
#    | FILTER BUOP V          # filter
#    | COUNT BUOP V           # count
#    | SCANL1 ABOP V          # scanl1
#    | ZIPWITH ABOP V V       # zipwith
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
# V -> a_1 | a_2              # inputs
#    | x_1 | x_2 | ... | x_T  # variables
# AUOP -> +1 | -1 | *2 | /2 | *(-1) | **2 | *3 | /3 | *4 | /4
# BUOP -> >0 | <0 | %2==0 | %2==1
# ABOP -> + | * | MIN | MAX

# abstract classes to implement
from synthrl.language.abstract import Node
from synthrl.language.abstract import Tree
from synthrl.language.abstract import UndefinedSemantics
from synthrl.utils.decoratorutils import classproperty
from synthrl.value import Integer
from synthrl.value import IntList

# maximum length of sequence
T = 5
# symbol L
class ListLang(Tree):
  
  __slots__ = [
    'input_types',
    'instructions',
    'output_type',
  ]

  def __init__(self, input_types, output_type=None):
    super(ListLang, self).__init__()
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
    if len(self.input_types) > 2:
      raise ValueError('Expecting at most 2 inputs, but {} inputs are given.'.format(len(self.input_types)))
    if self.input_types[0] != IntList:
      raise ValueError('First argument must be IntList, but {} is given.'.format(self.input_types[0].__name__))
    self.output_type = Integer if output_type == int or output_type == Integer else IntList if output_type == list or output_type == IntList else None
    self.instructions = [InstNode(parent='self') for _ in range(T)]

  @property
  def production_space(self):
    last_type = self.input_types[-1]
    var_types = {'a_{}'.format(i + 1): ty for i, ty in enumerate(self.input_types)}
    for i, inst in enumerate(self.instructions):
      self.hole, space = inst.production_space(loc=i, last_type=last_type, return_type=self.output_type, var_types=var_types)
      if len(space) > 0:
        break
      last_type = inst.return_type
      var_types['v_{}'.format(i + 1)] = inst.return_type
    return space

  def interprete(self, inputs):
    raise NotImplementedError

  def pretty_print(self):
    raise NotImplementedError

  @classproperty
  @classmethod
  def tokens(cls):
    raise NotImplementedError

# symbol I
class InstNode(Node):

  # dictionary has properties of tokens
  # TOKENS = {token: [options, # of parameters, return type]}
  TOKENS = {
    'map':     ['AUOP'  , 'ONE', 'LIST'],
    'filter':  ['BUOP'  , 'ONE', 'LIST'],
    'count':   ['BUOP'  , 'ONE', 'INT' ],
    'scanl1':  ['ABOP'  , 'ONE', 'LIST'],
    'zipwith': ['ABOP'  , 'TWO', 'LIST'],
    'head':    ['NOOPT' , 'ONE', 'INT' ],
    'last':    ['NOOPT' , 'ONE', 'INT' ],
    'minimum': ['NOOPT' , 'ONE', 'INT' ],
    'maximum': ['NOOPT' , 'ONE', 'INT' ],
    'reverse': ['NOOPT' , 'ONE', 'LIST'],
    'sort':    ['NOOPT' , 'ONE', 'LIST'],
    'sum':     ['NOOPT' , 'ONE', 'INT' ],
    'take':    ['REQINT', 'TWO', 'LIST'],
    'drop':    ['REQINT', 'TWO', 'LIST'],
    'access':  ['REQINT', 'TWO', 'INT' ],
    'nop':     ['END'   , 'ONE', 'LIST'],
  }
  
  def production_space(self, loc=1, last_type=None, return_type=None, var_types={}):
    if self.data == 'hole':
      options = ['AUOP', 'BUOP', 'ABOP', 'NOOPT']
      types = []
      if return_type and last_type == return_type:
        options.append('END')
      if Integer in [e for _, e in var_types.items()]:
        options.append('REQINT')
      if loc == T and return_type:
        types.append('LIST' if return_type == IntList else 'Int')
      else:
        types.extend(['LIST', 'INT'])
      return self, [token for token, prop in self.TOKENS if prop[0] in options and prop[2] in types]
    else:
      raise NotImplementedError

  def production(self, rule):
    raise NotImplementedError

  def interprete(self):
    raise NotImplementedError

  def pretty_print(self):
    raise NotImplementedError

  @classproperty
  @classmethod
  def tokens(cls):
    return list(cls.TOKENS.keys())

  @property
  def return_type(self):
    return IntList if self.TOKENS[self.data][2] == 'LIST' else Integer
