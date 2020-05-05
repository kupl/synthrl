if __name__ != '__main__':
  raise ImportError('This file cannot be imported.')

from itertools import cycle

from synthrl.language import ListLang

windmill = cycle('/' * 10000 + '-' * 10000 + '\\' * 10000 + '|' * 10000)

# unit testing
pool = [
  {
    'pgm': '''a_1 <- [int]; a_2 <- int; x_1 <- ACCESS a_2 a_1;''',
    'io': [
      (([0, 1, 2], 2), 2),
      (([2, 3, 1], 1), 3)
    ]
  },
  {
    'pgm': '''a_1 <- [int]; a_2 <- int; x_1 <- SORT a_1; x_2 <- DROP a_2 x_1; x_3 <- HEAD x_2''',
    'io': [
      (([0, 1, 2], 2), 2),
      (([2, 3, 1], 1), 2)
    ]
  },
]

for program_count, e in enumerate(pool):
  try:
    pgm = ListLang.parse(e['pgm'])
    for i, o in e['io']:
      assert (pgm.interprete(i).get_value() == o)
      print('Testing Program {}...{}'.format(program_count, next(windmill)), end='\r')
  except AssertionError:
    print('Testing failed in program {}               '.format(program_count))
  else:
    print('Program {} passed testing.           '.format(program_count))
