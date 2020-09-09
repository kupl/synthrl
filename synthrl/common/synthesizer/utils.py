
def check(program, ioset):
  for input, output in ioset:
    if program(input) != output:
      return False
  return True
