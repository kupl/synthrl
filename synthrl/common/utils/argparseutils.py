import argparse
import re

def add_help(parser):
  parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')

class regex:
  def __init__(self, pat, type=str):
    self.pat = re.compile(pat)
    self.type = type
  def __call__(self, arg):
    if not self.pat.match(arg):
      raise argparse.ArgumentTypeError
    return self.type(arg)
