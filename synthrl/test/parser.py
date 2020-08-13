from synthrl.common.utils.argparseutils import add_help
from synthrl.common.utils.argparseutils import regex

def add_arguments(parser):

  required_positional = parser.add_argument_group('required named arguments')

  # Add help.
  add_help(parser)
  
  # Environment arguments.
  required_positional.add_argument('--language', required=True, type=str, choices=['bitvec'], help='DSL to use.')
  required_positional.add_argument('--dataset', required=True, type=regex(r'.*\.json'), metavar='{*.json}', help='Source of oracle.')
  
  # Synthesizer arguments.
  parser.add_argument('--synth', choices=['rand'], default='rand', type=str, help='Choose synthesizer agent.')

  # Verifier arguments.
  parser.add_argument('--veri', choices=['rand'], default='rand', type=str, help='Choose verifier agent.')
