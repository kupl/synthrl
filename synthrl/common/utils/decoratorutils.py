
class classproperty:

  def __init__(self, getter):
    self.getter = getter if isinstance(getter, (classmethod, staticmethod)) else classmethod(getter)
  
  def __get__(self, instance, owner):
    return self.getter.__get__(instance, owner)()
