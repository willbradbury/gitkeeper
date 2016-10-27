""" Model generation class and model interface class
    (c) Alex Wang, Shivaal Roy, Will Bradbury
"""

class ModelFactory(type):
  """Metaclass to create models from a model name"""
  def __new__(cls, model, holdout, repo):
    # For now, we will not create any real models.
    # Just return an instance of the abstract class Model
    return Model()

class Model(object):
  """defines all the functions needed of a model."""
  def __init__(self): pass
  def train(self, verbosity=0): pass
  def test(self, verbosity=0): pass
