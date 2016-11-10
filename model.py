""" Model generation class and model interface class
    (c) Alex Wang, Shivaal Roy, Will Bradbury
"""

def model_factory(model, repo, v):
    """Return a new instance of the model specified by |model|"""
    return Model._registry[model](repo=repo, v=v)

class ModelRegistry(type):
  """Metaclass to keep track of defined models."""
  def __init__(cls, name, bases, dct):
    print 'registering %s' % (name,)
    if not hasattr(cls, '_registry'):
      cls._registry = {}
    cls._registry[name] = cls
    super(ModelRegistry, cls).__init__(name, bases, dct)

class Model(object):
  __metaclass__ = ModelRegistry
