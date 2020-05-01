from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from synthrl.utils.decoratorutils import classproperty

class Tree:
  def __init__(self, data:str='hole', children:dict={}, parent:'Tree'=None) -> None:
    self.data = data
    self.children = children
    self.parent = parent
    
  def production_space(self) -> List[str]:
    # returns a list of possible production rules
    raise NotImplementedError

  def production(self, rule:str) -> 'Tree':
    # gets one production rule to apply
    # returns self
    raise NotImplementedError

  def interprete(self, *args, **kwargs) -> Any:
    # gets needed information
    # returns an executed result of program
    raise NotImplementedError

  def pretty_print(self, file=None, *args, **kwargs) -> None:
    # print a tree node
    raise NotImplementedError

  @classproperty
  @classmethod
  def tokens(cls) -> List[str]:
    # returns a list of tokens
    raise NotImplementedError

  @property
  def spec(self) -> Dict[str, Any]:
    # returns a dictionary that contains all information to create hole
    raise NotImplementedError
  