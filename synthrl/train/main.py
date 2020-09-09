from pathlib import Path
import torch
import torch.multiprocessing as mp

from synthrl.common.utils import rmtree
from synthrl.train.parser import add_arguments

def main(parser, argv):

  add_arguments(parser)
  args = parser.parse_args(argv)
  print(args)

  # # Check arguments.
  # if args.workers < 3:
  #   raise ValueError('Workers should be 3 or greater than 3.')
  
  # # Make workdir.
  # workdir = Path(args.workdir)
  # if workdir.exists():
  #   if not rmtree(workdir, confirmed=args.del_existing_workdir):
  #     print('Directory "{}" already exist. Data could be overwritten.'.format(workdir))
  # workdir.mkdir(parents=True, exist_ok=True)

  # # Get device setting.
  # device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

  
