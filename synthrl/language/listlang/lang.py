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

from synthrl.language.abstract import Node
from synthrl.language.abstract import Tree
from synthrl.language.abstract import UndefinedSemantics
from synthrl.language.abstract import WrongProductionException
from synthrl.utils.decoratorutils import classproperty
from synthrl.utils.exceptionutils import UnexpectedException
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
    if len(self.input_types) > 2:
      raise ValueError('Expecting at most 2 inputs, but {} inputs are given.'.format(len(self.input_types)))
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
  
  def production_space(self, loc=1, last_type=None, return_type=None, var_types={}):
    
    # if the node should be filled
    if self.data == 'hole':

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
    # returns the return type of instruction
    return IntList if self.TOKENS[self.data][2] == 'LIST' else Integer

class VarNode(Node):

  def __init__(self, type=IntList, *args, **kwargs):

    super(VarNode, self).__init__(*args, **kwargs)

    # store the desiring type
    self.type = type

class AUOPNode(Node):
  pass

class BUOPNode(Node):
  pass

class ABOPNode(Node):
  pass
