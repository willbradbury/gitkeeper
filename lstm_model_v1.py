""" First pass at LSTM based model implementation.
    (c) Alex Wang, Shivaal Roy, Will Bradbury"""

import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import json, util, random
import model

class LSTMModel(model.Model):
  def __init__(self, repo, v=1):
    self.v = v
    self.diff_tokenizer = DiffTokenizer(v=v)
    self.repo_tokenizer = RepoTokenizer(v=v)
    self.token_embedder = TokenEmbedder(1000, v=v)
    self.repo = repo

  def train(self):
    util.log(self.v, 1, "beginning training on " + self.repo.name)
    self.train_set = list(self.embedder.embed(self.repo_tokenizer.tokenize(self.repo)))
    self.dev_set = list(self.embedder.embed(self.diff_tokenizer.tokenize(self.repo, 'train')))
    self.lstm_trainer = LSTMTrainer(self.train_set, self.dev_set)
    self.lstm_trainer.get_trainer().run()
    util.log(self.v, 1, "mean training accuracy: " + str(score))

  def test(self):
    util.log(self.v, 1, "beginning testing on " + self.repo.name)
    pass
    util.log(self.v, 1, "mean testing accuracy: " + str(score))

class DiffTokenizer(object):
  """A class to tokenize diff files"""
  def __init__(self, v):
    self.v = v

  def tokenize(self, repo, subset):
    util.log(self.v, 3, "tokenizing diffs in "+repo.name+"/"+subset)
    # tokenize using the yield operator
    return

class RepoTokenizer(object):
  """A class to tokenize repo files"""
  def __init__(self, v):
    self.v = v

  def tokenize(self, repo):
    util.log(self.v, 3, "tokenizing repo " + repo.name)
    # tokenize using the yield operator
    return

class TokenEmbedder(object):
  """A class to turn tokens into token ids"""
  def __init__(self, embed_size, v):
    self.v = v
    self.size = embed_size # number of total IDs to make

  def embed(self, token_stream):
    util.log(self.v, 3, "embedding stream")
    for token in token_stream:
      yield hash(token) % self.size #TODO: CHANGE THIS!!!
    return
