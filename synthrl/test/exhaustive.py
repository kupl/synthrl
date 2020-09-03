import logging
import numpy as np
import time

from synthrl.agent.agent import Agent

logger = logging.getLogger(__name__)

class Node:
  def __init__(self, data='hole', parent=None):
    self.data = data
    self.parent = parent
    self.children = None
    self._done = False
    self.children_done = None
  
  def take_action(self, actions=[]):
    if not self.children:
      self.children = {}
      self.children_done = {}
      for action in actions:
        self.children[action] = Node(action, self)
        self.children_done[action] = False
    _actions = [key for key, done in self.children_done.items() if not done]
    if len(_actions) == 0:
      logger.warning('All possible actions are explored.')
      _actions = actions
    return np.random.choice(_actions)

  def signal_child_done(self, child=None, done=False):
    if not done:
      return
    self.children_done[child] = True
    for _, done in self.children_done.items():
      if not done:
        return
    self._done = True
    if self.parent:
      self.parent.signal_child_done(self.data, True)

  @property
  def done(self):
    return self._done

  @done.setter
  def done(self, done):
    self._done = done
    self.parent.signal_child_done(self.data, done)

  def __repr__(self):
    return self.data

  def __str__(self):
    return repr(self)

class Tree:
  def __init__(self):
    self.root = Node('root')
    self.last = self.root
  
  def reset(self):
    if self.root != self.last:
      self.last.done = True
    self.last = self.root

  def take_action(self, actions=[]):
    action = self.last.take_action(actions)
    self.last = self.last.children[action]
    return action

class ExhaustiveAgent(Agent):
  def __init__(self, mode=None):
    self.tree = Tree()
    if mode in 'synthesizer':
      self.mode = 1
    elif mode in 'verifier':
      self.mode = 2
    else:
      raise ValueError('Invalid mode is given: {}'.format(mode))

  def take(self, state=None, action_space=[]):
    if state[self.mode].is_hole():
      self.tree.reset()
    action = self.tree.take_action(action_space)
    return action
    
  def reset(self):
    self.tree = Tree()
    