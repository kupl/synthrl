from synthrl.language import ListLanguage

program = ListLanguage()
iter = 0
while True:
  
  node, space = program.production_space()
  if len(space) == 0:
    break

  iter += 1
  print('-- Iter {} --'.format(iter))
  print('* Program')
  program.pretty_print()
  
  print('* Select from candidates: ')
  for i, s in enumerate(space):
    print(i, s)
  selection = int(input())
  node.production(space[selection])
  print()

print('-- Final program --')
program.pretty_print()
