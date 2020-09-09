import argparse
import sys

# pylint: disable=no-name-in-module,no-member
from synthrl.train.main import main as trainer
from synthrl.test.main import main as tester
from synthrl.common.utils import Color
import synthrl

def main(argv):

  # Main parser.
  parser = argparse.ArgumentParser()
  parser.add_argument('--no-color', action='store_true', help='Disable color.')
  # Add mode
  mode = parser.add_subparsers(title='mode', dest='mode', help='Choose one of the followings.')
  train_parser = mode.add_parser(name='train', help='Train mode.', add_help=False)
  test_parser = mode.add_parser(name='test', help='Test mode.', add_help=False)

  # Parse mode argument.
  args, unknown = parser.parse_known_args(argv)
  if args.no_color:
    Color.disable()

  print(f'{Color.BOLD}{Color.GREEN}SynthRL{Color.END} ver.{synthrl.__version__}')
  print()

  # If train mode.
  if args.mode == 'train':
    trainer(train_parser, unknown)

  # If test mode.
  elif args.mode == 'test':
    tester(test_parser, unknown)

  # If mode not passed.
  else:
    parser.parse_args(['--help'])

if __name__ == '__main__':
  main(sys.argv[1:])
