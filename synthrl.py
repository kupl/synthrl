import argparse
import sys

from synthrl.train.main import main as trainer
from synthrl.test.main import main as tester

def main(argv):

  # Main parser.
  parser = argparse.ArgumentParser(prog='python synthrl.py')
  # Add mode
  mode = parser.add_subparsers(title='mode', dest='mode', help='Choose one of the followings.')
  mode.required = True
  train_parser = mode.add_parser(name='train', help='Train mode.', add_help=False)
  test_parser = mode.add_parser(name='test', help='Test mode.', add_help=False)

  # Parse mode argument.
  args, unknown = parser.parse_known_args(argv)

  # If train mode.
  if args.mode == 'train':
    trainer(train_parser, unknown)

  # If test mode.
  else: # args.mode == 'test'
    tester(test_parser, unknown)

if __name__ == '__main__':
  main(sys.argv[1:])
