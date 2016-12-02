""" First pass at LSTM based model implementation.
    (c) Alex Wang, Shivaal Roy, Will Bradbury"""

# chainer imports
from __future__ import division
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

# sklearn imports
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

# gitkeeper imports
from lstm_trainer import LSTMTrainer
import json, util, random, model

class RNN(Chain):
  """3 layer RNN with one LSTM layer and a 1000 neuron embedding."""
  def __init__(self):
    super(RNN, self).__init__(
        embed = L.EmbedID(1000, 100), # word embedding
        mid = L.LSTM(100, 50), # the first LSTM layer
        out = L.Linear(50, 1000), # the feed-forward output layer
    )

  def reset_state(self):
    self.mid.reset_state()

  def __call__(self, cur_word):
    """Predict the next word given the |cur_word| id."""
    return self.out(self.mid(self.embed(cur_word)))

class LSTMModel(model.Model):
  def __init__(self, repo, v=1):
    self.v = v
    self.rnn_layout = RNN
    self.file_tokenizer = FileTokenizer(v=v)
    self.diff_tokenizer = DiffTokenizer(self.file_tokenizer, v=v)
    self.repo_tokenizer = RepoTokenizer(self.file_tokenizer, v=v)
    self.embedder = TokenEmbedder(embed_size=1000, cap=100, v=v)
    self.extractor = SimpleExtractor(v=v)
    self.repo = repo
    self.clf = SVC()

  def train(self):
    util.log(self.v, 1, "beginning training on " + self.repo.name)

    # tokenize the repo and the training diffs
    self.train_set = np.array(list(self.embedder.embed(self.repo_tokenizer.tokenize(self.repo))), dtype=np.int32)
    self.dev_set = np.array(list(self.embedder.embed(self.diff_tokenizer.tokenize(self.repo))), dtype=np.int32)

    # train the rnn on the repo, reporting dev error along the way
    self.lstm_trainer = LSTMTrainer(self.rnn_layout, self.train_set, self.dev_set, v=self.v)
    self.lstm_trainer.get_trainer().run()

    # learn svm on the perplexity feature
    x,y = np.array([]), np.array([])
    for i,pid in enumerate(self.repo.getExamples(inTraining=True)):
      diff_f = self.repo.getDiffFile(pid, inTraining=True)
      x = np.append(x, self.get_perplexity(diff_f))

      meta_f = self.repo.getMetaFile(pid, inTraining=True)
      meta_json = json.load(meta_f)
      y = np.append(y,self.extractor.label(meta_json))
    X = x.reshape(x.size,1)
    self.clf.fit(X,y)
    score = self.clf.score(X,y)
      
    util.log(self.v, 1, "mean training accuracy: " + str(score))

  def test(self):
    util.log(self.v, 1, "beginning testing on " + self.repo.name)
    x,y = np.array([]), np.array([])
    for i,pid in enumerate(self.repo.getExamples(inTraining=False)):
      diff_f = self.repo.getDiffFile(pid, inTraining=False)
      x = np.append(x, self.get_perplexity(diff_f))

      meta_f = self.repo.getMetaFile(pid, inTraining=False)
      meta_json = json.load(meta_f)
      y = np.append(y,self.extractor.label(meta_json))

    X = x.reshape(x.size,1)
    score = self.clf.score(X,y)
    util.log(self.v, 1, "mean testing accuracy: " + str(score))

  def get_perplexity(self, diff):
    """computes the loss when trying to predict the next token from each token
    in |diff|."""
    diff_tokens = list(self.embedder.embed(self.file_tokenizer.tokenize(diff), wrap=False))
    diff_tokens = diff_tokens[:1000]
    return self.lstm_trainer.compute_perplexity_slow(diff_tokens)

class FileTokenizer(object):
  """A class to tokenize files"""
  def __init__(self, v):
    self.v = v

  def tokenize(self, f):
    for char in f.read():
      yield char

class DiffTokenizer(object):
  """A class to tokenize a collection of diffs using a file tokenizer"""
  def __init__(self, ft, v):
    self.v = v
    self.ft = ft
    self.extractor = SimpleExtractor(v=v)

  def tokenize(self, repo, inTraining=True):
    # tokenize using the yield operator
    util.log(self.v, 3, "tokenizing diffs in "+repo.name+"/"+("train" if inTraining else "test"))

    for i,pid in enumerate(repo.getExamples(inTraining=inTraining)):
      meta_f = repo.getMetaFile(pid, inTraining=inTraining)
      meta_json = json.load(meta_f)
      # validation accuracy should only include accepted diffs
      if self.extractor.label(meta_json) != 1: continue

      diff_f = repo.getDiffFile(pid, inTraining=inTraining)
      for token in self.ft.tokenize(diff_f):
        yield token

class RepoTokenizer(object):
  """A class to tokenize an entire repo using a file tokenizer"""
  def __init__(self, ft, v):
    self.v = v
    self.ft = ft

  def tokenize(self, repo):
    util.log(self.v, 3, "tokenizing repo " + repo.name)
    # tokenize using the yield operator
    for fileName in repo.getDirList():
      util.log(self.v, 3, "Tokenizing fileName: "+ fileName)
      f = open(fileName)
      for token in self.ft.tokenize(f):
        yield token

class TokenEmbedder(object):
  """A class to turn tokens into token ids"""
  def __init__(self, embed_size, v, cap=5000):
    self.v = v
    self.size = embed_size # number of total IDs to make
    self.cap = cap

  def embed(self, token_stream, wrap=False):
    util.log(self.v, 3, "embedding stream")
    embeds = 0
    for token in token_stream:
      if wrap: yield np.array(min(ord(token), self.size-1))
      else: yield min(ord(token), self.size-1)
      embeds += 1
      if embeds > self.cap: break

class SimpleExtractor(object):
  def __init__(self, v=1):
    self.v = v

  def extract(self, diff_f, meta_json):
    return None

  def label(self, meta_json):
    if meta_json['merged']: return 1
    return -1
