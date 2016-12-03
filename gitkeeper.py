""" Primary commandline interface for using gitkeeper
    (c) Alexander Wang, Shivaal Roy, Will Bradbury
"""
import sys
import util
from model import Model, model_factory
#from baseline_model import BaselineModel
from lstm_model_avg import LSTMModel

holdout_fraction = 0.2
verbosity = 2

def usage():
  print "gitkeeper (c) Alex Wang, Shivaal Roy, Will Bradbury"
  print "usage: python gitkeeper REPOSITORY [REPOSITORY] ..."
  print "e.g. python gitkeeper rust-lang/rust go-lang/go"

def main():
  if '--help' in sys.argv or '-h' in sys.argv:
    usage()
    return

  # Iterate through requested repositories
  for repo in sys.argv[1:]:
    # download it or find it locally
    rp = util.download(repo, v=verbosity)

    # build all the models
    for model in Model._registry:
      util.log(verbosity, 2, "training model " + model)
      m = model_factory(model, repo=rp, v=verbosity)
      m.train()
      util.log(verbosity, 2, "testing model " + model)
      m.test()

if __name__ == '__main__' : main()
