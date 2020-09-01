from synthrl.common.utils.argparseutils import add_help
from synthrl.common.utils.argparseutils import regex

def add_arguments(parser):

  required_positional = parser.add_argument_group('required named arguments')

  # Add help.
  add_help(parser)
  
  # Environment arguments.
  required_positional.add_argument('--setting', required=True, type=regex(r'.*\.json'), metavar='{*.json}', help='Test settings.')
  
  # Synthesizer arguments.
  required_positional.add_argument('--synth', required=True, type=str, help='Choose synthesizer agent.')
  parser.add_argument('--synth-args', nargs='*', default=[], metavar='param=arg', type=str, help='Synthesizer specific arguments.')
  required_positional.add_argument('--synth-func', required=True, type=str, help='Choose a function to use.')
  parser.add_argument('--synth-func-args', nargs='*', default=[], metavar='param=arg', type=str, help='Function sepecific arguments.')
  required_positional.add_argument('--synth-max-move', required=True, type=int, help='Max move for synthesizer.')

  # Verifier arguments.
  parser.add_argument('--veri', choices=['rand'], default='rand', type=str, help='Choose verifier agent.')
