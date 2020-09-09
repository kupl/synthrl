from synthrl.common.utils.argparseutils import add_help
from synthrl.common.utils.argparseutils import regex

def add_arguments(parser):

  # Add help.
  add_help(parser)
  
  # Environment arguments.
  env_args = parser.add_argument_group('environment arguments')
  env_args.add_argument('--setting', required=True, type=regex(r'.*\.json'), metavar='{*.json}', help='Test settings.')
  
  # Synthesizer arguments.
  synth_args = parser.add_argument_group('synthesizer arguments')
  synth_args.add_argument('--synth', required=True, type=str, metavar='<agent>', help='Choose synthesizer agent.')
  synth_args.add_argument('--synth-args', nargs='*', default=[], metavar='param=arg', type=str, help='Chosen synthesizer specific arguments.')
  synth_args.add_argument('--synth-func', required=True, type=str, metavar='<function>', help='Choose a function to use.')
  synth_args.add_argument('--synth-func-args', nargs='*', default=[], metavar='param=arg', type=str, help='Chosen function sepecific arguments.')
  synth_args.add_argument('--synth-max-move', required=True, type=int, metavar='<int>', help='Max move for synthesizer.')

  # Verifier arguments.
  veri_args = parser.add_argument_group('verifier arguments')
  veri_args.add_argument('--veri', required=True, type=str, metavar='<agent>', help='Choose verifier agent.')
  veri_args.add_argument('--veri-args', nargs='*', default=[], metavar='param=arg', type=str, help='Chosen verifier specific arguments.')
  veri_args.add_argument('--veri-func', required=True, type=str, metavar='<function>', help='Choose a function to use.')
  veri_args.add_argument('--veri-func-args', nargs='*', default=[], metavar='param=arg', type=str, help='Chosen function sepecific arguments.')
  veri_args.add_argument('--testing', required=True, type=str, metavar='<testing>', help='Choose a testing methodo to use.')
  veri_args.add_argument('--testing-args', nargs='*', default=[], metavar='param=arg', type=str, help='Chosen testing method specific arguments.')
  veri_args.add_argument('--veri-max-move', required=True, type=int, metavar='<int>', help='Max move for verifier.')
