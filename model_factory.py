""" Model generation class and model interface class
    (c) Alex Wang, Shivaal Roy, Will Bradbury
"""
import baseline_model

model_map = {
    'baseline' : baseline_model.BaselineModel
}

class ModelFactory(type):
  """Metaclass to create models from a model name"""
  def __new__(cls, model, repo, v):
    # For now, we will not create any real models.
    # Just return an instance of the abstract class Model
    return model_map[model](repo=repo, v=v)
