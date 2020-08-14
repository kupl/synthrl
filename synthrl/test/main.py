import argparse

from synthrl.test.parser import add_arguments
from synthrl.common.environment import Dataset

def main(parser, argv):

  # Parse arguments.
  add_arguments(parser)
  args = parser.parse_args(argv)
  print(args)

  # Load dataset.
  dataset = Dataset.from_json(args.setting)
  language = dataset.language

  # For each data in dataset.
  for oracle, ioset in dataset:

    while True:

      # Create synthesizer.
      synthesizer = None
      prog = synthesizer.synthesize(ioset)

      # Create verifier.
      verifier = None
      distinguishing_input = verifier.verifier(prog, ioset)

      if distinguishing_input is None:
        break
      desired_output = oracle(distinguishing_input)

      ioset.add(distinguishing_input, desired_output)
  