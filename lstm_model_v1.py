""" First pass at LSTM based model implementation.
    (c) Alex Wang, Shivaal Roy, Will Bradbury"""

from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from lstm_trainer import LSTMTrainer
import json, util, random, model
import numpy as np

class LSTMModel(model.Model):
  def __init__(self, repo, v=1):
    self.v = v
    self.file_tokenizer = FileTokenizer(v=v)
    self.diff_tokenizer = DiffTokenizer(self.file_tokenizer, v=v)
    self.repo_tokenizer = RepoTokenizer(self.file_tokenizer, v=v)
    self.token_embedder = TokenEmbedder(embed_size=1000, v=v)
    self.repo = repo

  def train(self):
    util.log(self.v, 1, "beginning training on " + self.repo.name)

    # tokenize the repo and the training diffs
    self.train_set = [Variable(i) \
        for i in self.embedder.embed(self.repo_tokenizer.tokenize(self.repo))]
    self.dev_set = [Variable(i) \
        for i in self.embedder.embed(self.diff_tokenizer.tokenize(self.repo))]

    # train the rnn on the repo, reporting dev error along the way
    self.lstm_trainer = LSTMTrainer(self.train_set, self.dev_set)
    self.lstm_trainer.get_trainer().run()

    # learn svm on the perplexity feature
    x,y = np.array([]), np.array([])
    for i,pid in enumerate(self.repo.getExamples(inTraining=True)):
      diff_f = self.repo.getDiffFile(pid, inTraining=True)
      x = np.append(x, self.get_perplexity(diff_f))

      meta_f = self.repo.getMetaFile(pid, inTraining=True)
      meta_json = json.load(meta_f)
      y = np.append(y,self.extractor.label(meta_json), axis=0)
    X = np.transpose(x) # may not be necessary... TODO: check
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
      y = np.append(y,self.extractor.label(meta_json), axis=0)

    X = np.transpose(x) # necessary? TODO: check
    score = self.clf.score(X,y)
    util.log(self.v, 1, "mean testing accuracy: " + str(score))

  def get_perplexity(self, diff):
    """computes the loss when trying to predict the next token from each token
    in |diff|."""
    diff_tokens = list(self.embedder.embed(self.file_tokenizer.tokenize(diff)))
    volatile_tokens = [Variable(tok_id, volatile='on') for tok_id in diff_tokens]
    return self.lstm_trainer.compute_loss(volatile_tokens)

class FileTokenizer(object):
  """A class to tokenize files"""
  def __init__(self, v):
    self.v = v

  def tokenize(self, f):
    # tokenize using the yield operator
    return

class DiffTokenizer(object):
  """A class to tokenize a collection of diffs using a file tokenizer"""
  def __init__(self, ft, v):
    self.v = v
    self.ft = ft

  def tokenize(self, repo, subset):
    util.log(self.v, 3, "tokenizing diffs in "+repo.name+"/"+subset)
    # tokenize using the yield operator
    return

class RepoTokenizer(object):
  """A class to tokenize an entire repo using a file tokenizer"""
  def __init__(self, ft, v):
    self.v = v
    self.ft = ft

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
      yield np.array(hash(token) % self.size) #TODO: CHANGE THIS!!!
