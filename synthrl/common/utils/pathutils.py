from pathlib import Path
import sys

def rmtree(path, confirmed=False):
  path = Path(path)

  if not confirmed:
    answer = 'x'
    while answer.lower() not in ['y', 'n', '']:
      answer = input('Remove "{}" (y/N)? '.format(path)).strip()
    if answer.lower() in ['n', '']:
      return False

  for child in path.glob('*'):
    if child.is_file():
      child.unlink()
    else:
      rmtree(child, confirmed=True)
  path.rmdir()
  return True
