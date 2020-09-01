import argparse

from synthrl.common.environment import Dataset
from synthrl.common.synthesizer.exception import SynthesisFailed
from synthrl.common.utils import mkdict
from synthrl.common.utils import Color
from synthrl.test.parser import add_arguments
import synthrl.common.function as function_module
import synthrl.common.language as language_module
import synthrl.common.synthesizer as synthesizer_module
import synthrl.common.verifier as verifier_module

def main(parser, argv):

  # Parse arguments.
  add_arguments(parser)
  args = parser.parse_args(argv)

  # Load dataset.
  print(f'{Color.BOLD}Dataset loading.{Color.END}')
  dataset = Dataset.from_json(args.setting)
  language = getattr(language_module, dataset.language)
  print(f'  language: {dataset.language}')
  print(f'    length: {len(dataset)}')
  print(f'{Color.BOLD}{Color.BLUE}Dataset succesfully loaded.{Color.END}')
  print()

  # Synthesizer info.
  print(f'{Color.BOLD}Creating synthesizer.{Color.END}')
  synth_func_class = getattr(function_module, args.synth_func)
  synth_func_args = mkdict(args.synth_func_args)
  print(f'   {Color.UNDERLINE}Function{Color.END}: {Color.YELLOW}{args.synth_func}{Color.END}({", ".join([f"language={dataset.language}"] + args.synth_func_args)})')
  synth_func = synth_func_class(language=language, **synth_func_args)

  synthesizer_class = getattr(synthesizer_module, args.synth)
  synth_args = mkdict(args.synth_args)
  print(f'Synthesizer: {Color.YELLOW}{args.synth}{Color.END}({", ".join([f"language={dataset.language}", f"function={Color.UNDERLINE}Function{Color.END}"] + args.synth_args)})')
  synthesizer = synthesizer_class(language=language, function=synth_func, **synth_args)

  print(f'{Color.BOLD}{Color.BLUE}Synthesizer succesfully created.{Color.END}')
  print()

  # Verifier info.
  print(f'{Color.BOLD}Creating verifier.{Color.END}')

  # verifier_class = getattr(verifier_module, args.veri)

  print(f'{Color.BOLD}{Color.BLUE}Verifier succesfully created.{Color.END}')
  print()


  # For each data in dataset.
  print(f'{Color.BOLD}Start main procedure.{Color.END}')
  for i, (oracle, ioset) in enumerate(dataset):
    print(f'Program # {i + 1}')
    print('----- oracle -----')
    oracle.pretty_print()
    print('----- result -----')

    try:
      iteration = 0
      while True:
        iteration += 1

        # Synthesize program.
        prog = synthesizer.synthesize(ioset, args.synth_max_move)
        prog.pretty_print()

        # Make distinguising input.
        verifier = None
        distinguishing_input = verifier.verifier(prog, ioset)

        if distinguishing_input is None:
          break
        desired_output = oracle(distinguishing_input)

        ioset.add(distinguishing_input, desired_output)
        
    except SynthesisFailed:
      print(f'{Color.BOLD}{Color.RED}Failed:{Color.END} iteration # {iteration}')
    finally:
      print('----- ioset ------')
      print(ioset)
    
    print()
  