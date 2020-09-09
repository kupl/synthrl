
def check(candidate, program, ioset, testing):
  for input, output in ioset:
    if program(input) != output:
      return None
  return testing(candidate, program)  
