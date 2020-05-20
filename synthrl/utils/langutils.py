from synthrl.language import ListLang
from synthrl.language.abstract import Tree

def Language(language):
  if isinstance(language, type) and issubclass(language, Tree):
    return language
  elif isinstance(language, str):
    if language == 'ListLang':
      return ListLang
    else:
      raise ValueError('Unkwon language: {}'.format(language))
  else:
    raise TypeError('Language() arg 1 must be a class or string.')
