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
from sklearn.metrics import recall_score, precision_score, accuracy_score

# gitkeeper imports
from lstm_trainer import LSTMTrainer
import json, util, random, model, re, collections

class RNN(Chain):
  """4 layer RNN with 2 LSTM layers and a 1000 neuron embedding."""
  def __init__(self, lstm_width=50, train=True):
    super(RNN, self).__init__(
        embed = L.EmbedID(1000, lstm_width), # word embedding
        l1 = L.LSTM(lstm_width, lstm_width), # the first LSTM layer
        l2 = L.LSTM(lstm_width, lstm_width), # the second LSTM layer
        out = L.Linear(lstm_width, 1000), # the feed-forward output layer
    )
    self.train = train

  def reset_state(self):
    self.l1.reset_state()
    self.l2.reset_state()

  def __call__(self, cur_word):
    """Predict the next word given the |cur_word| id."""
    h0 = self.embed(cur_word)
    h1 = self.l1(F.dropout(h0, train=self.train))
    h2 = self.l2(F.dropout(h1, train=self.train))
    out = self.out(F.dropout(h2, train=self.train))
    return out

class LSTMModel(model.Model):
  def __init__(self, repo, v=1):
    self.v = v
    
    # Model parameters (along with lstm_width above)
    self.epochs = 1 # number of runs through the entire data during training
    self.offsets = 3 # number of pointers in the data during training
    self.bprop_depth = 35 # how many token are rememebered by the rnn
    embed_size = 1000 # number of allowable tokens/characters
    tokenization_cap = 10000 # how many tokens are read from the repo
    self.test_cap = 100 # how many tokens are read from each diff (< token cap)

    self.rnn_layout = RNN # set the layout
    self.extractor = SimpleExtractor(v=v)
    self.file_tokenizer = FileTokenizer(v=v)
    self.diff_tokenizer = DiffTokenizer(self.file_tokenizer, v=v)
    self.repo_tokenizer = RepoTokenizer(self.file_tokenizer, v=v)
    self.repo = repo
    self.embedder = TokenEmbedder(embed_size=embed_size,
                                  cap=tokenization_cap,
                                  corpus_itr=chain_itrs(\
                                      self.repo_tokenizer.tokenize(self.repo),
                                      self.diff_tokenizer.tokenize(self.repo)),
                                  v=v)
    self.clf = SVC()

  def train(self):
    util.log(self.v, 1, "beginning training on " + self.repo.name)

    # tokenize the repo and the training diffs
    self.train_set = np.array(\
      list(self.embedder.embed(chain_itrs(\
                        self.repo_tokenizer.tokenize(self.repo),
                        self.diff_tokenizer.tokenize(self.repo)))),
      dtype=np.int32)
    self.dev_set = np.array(\
      list(self.embedder.embed(self.diff_tokenizer.tokenize(\
                        self.repo,
                        inTraining=False))),
      dtype=np.int32)

    # train the rnn on the repo, reporting dev error along the way
    self.lstm_trainer = LSTMTrainer(self.rnn_layout,
                                    self.train_set,
                                    self.dev_set,
                                    epochs=self.epochs,
                                    offsets=self.offsets,
                                    bprop_depth=self.bprop_depth,
                                    v=self.v)
    self.lstm_trainer.get_trainer().run()

  def test(self):
    util.log(self.v, 1, "beginning testing on " + self.repo.name)
    in_sum, out_sum = 0,0
    for i,pid in enumerate(self.repo.getExamples(inTraining=False)):
      # determine if the pull was accepted or rejected
      meta_f = self.repo.getMetaFile(pid, inTraining=False)
      meta_json = json.load(meta_f)
      label = self.extractor.label(meta_json)

      # get diff perplexity using trained rnn
      diff_f = self.repo.getDiffFile(pid, inTraining=False)
      x = self.get_perplexity(diff_f)
      if x==0: continue # skip empty files

      # increment the appropriate sum and counter
      if label == 1:
        in_sum, in_count = in_sum + x, in_count + 1
      else:
        out_sum, out_count = out_sum + x, out_count + 1

    # compute overall statistics
    in_avg = in_sum/in_count
    out_avg = out_sum/out_count
    util.log(self.v, 1,\
        "testing: in_sum, out_sum=%f %f" % (in_sum, out_sum))

  def get_perplexity(self, diff):
    """computes the loss when trying to predict the next token from each token
    in |diff|."""
    diff_tokens = list(self.embedder.embed(self.file_tokenizer.tokenize(diff)))
    diff_tokens = diff_tokens[:self.test_cap]
    return self.lstm_trainer.compute_perplexity(np.array(diff_tokens, dtype=np.int32))

def chain_itrs(itr1, itr2):
    for i in itr1:
        yield i
    for i in itr2:
        yield i

class FileTokenizer(object):
  """A class to tokenize files"""
  punctuation = r'(\.|,|\\|/|#|!|\$|%|\^|&|\*|;|:|{|}|=|-|_|`|~|\(|\)| |\t|\n)'

  def __init__(self, v):
    self.v = v

  def tokenize(self, f):
    for line in f:
      arr = self.tokenizeLine(line)
      for token in arr:
        yield token
      if len(arr): yield '\n'

  def tokenizeLine(self, line):
    l =  re.split(self.punctuation, line)
    q = [x for x in l if x != '']
    return [x for x,y in [(z1,z2) for z1,z2 in zip(q,q[1:]) if not(z1==z2 and z1 in [' ', '\t', '\n'])]]


class DiffTokenizer(object):
  """A class to tokenize a collection of diffs using a file tokenizer"""
  def __init__(self, ft, v):
    self.v = v
    self.ft = ft
    self.extractor = SimpleExtractor(v=v)

  def tokenize(self, repo, inTraining=True):
    # tokenize using the yield operator
    util.log(self.v, 3, "tokenizing diffs in "+repo.name+"/"+\
            ("train" if inTraining else "test"))

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
  def __init__(self, embed_size, corpus_itr, v, cap=5000):
    self.v = v
    self.size = embed_size # number of total IDs to make
    self.cap = cap

    # Create embedding from embed_size most common characters
    most_common = collections.Counter(corpus_itr).most_common(self.size)
    self.embedding = {t[0]:i for i,t in enumerate(most_common)}

  def embed(self, token_stream, wrap=False):
    util.log(self.v, 3, "embedding stream")
    embeds = 0
    for token in token_stream:
      if wrap: yield np.array(self.embedding.get(token,self.size-1))
      else: yield self.embedding.get(token,self.size-1)
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
