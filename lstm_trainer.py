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

import util

class ParallelSequentialIterator(chainer.dataset.Iterator):
  def __init__(self, dataset, batch_size, repeat=True):
    self.dataset = dataset
    self.batch_size = batch_size
    self.epoch = 0
    self.is_new_epoch = False
    self.repeat = repeat
    self.offsets = [i*len(dataset) // batch_size for i in range(batch_size)]
    self.iteration = 0

  def __next__(self):
    length = len(self.dataset)
    if not self.repeat and self.iteration*self.batch_size >= length:
      raise StopIteration
    cur_words = self.get_words()
    self.iteration += 1
    next_words = self.get_words()

    epoch = self.iteration * self.batch_size // length
    self.is_new_epoch = self.epoch < epoch
    if self.is_new_epoch:
      self.epoch = epoch

    return list(zip(cur_words, next_words))

  def epoch_detail(self):
    return self.iteration * self.batch_size / len(self.dataset)

  def get_words(self):
    return [self.dataset[(offset + self.iteration) % len(self.dataset)]
              for offset in self.offsets]

  def serialize(self, serializer):
    self.iteration = serializer('iteration', self.iteration)
    self.epoch = serializer('epoch', self.epoch)

class BPTTUpdater(training.StandardUpdater):
  def __init__(self, train_iter, optimizer, bprop_len, device):
    super(BPTTUpdater, self).__init__(
      train_iter, optimizer, device=device)
    self.bprop_len = bprop_len

  def update_core(self):
    loss = 0
    train_iter = self.get_iterator('main')
    optimizer = self.get_optimizer('main')

    for i in range(self.bprop_len):
      batch = train_iter.__next__()
      x,t = self.converter(batch, self.device)

      loss += optimizer.target(Variable(x), Variable(t))

    optimizer.target.cleargrads()
    loss.backward()
    loss.unchain_backward()
    optimizer.update()

class LSTMTrainer(object):
  def __init__(self, layout, train, dev, epochs, offsets, bprop_depth, v):
    self.v = v
    self.epochs = epochs
    self.offsets = offsets
    self.bprop_depth = bprop_depth
    self.rnn = layout()
    self.model = L.Classifier(self.rnn)
    self.eval_model = self.model.copy()
    self.eval_model.predictor.train = False
    self.optimizer = optimizers.SGD()
    self.optimizer.setup(self.model)
    self.train_iter = ParallelSequentialIterator(train, self.epochs, repeat=True)
    self.dev_iter = ParallelSequentialIterator(dev, 1, repeat=False)

  def get_trainer(self):
    updater = BPTTUpdater(self.train_iter, self.optimizer, self.bprop_depth, -1)
    trainer = training.Trainer(updater, (self.epochs, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(self.dev_iter, self.eval_model,
      eval_hook=lambda _: self.eval_model.predictor.reset_state()))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch',
      'main/accuracy',
      'validation/main/accuracy']))
    return trainer

  def compute_perplexity(self, test_set):
    self.eval_model.predictor.reset_state()
    test_itr = ParallelSequentialIterator(test_set, 1, repeat=False)
    evaluator = extensions.Evaluator(test_itr, self.eval_model, device=-1)
    result = evaluator()
    if not result: return 1
    util.log(self.v,3,result)
    return np.exp(float(result['main/loss'])/len(test_set))

  def compute_perplexity_slow(self, test_set):
    loss = 0
    for cur_word, next_word in zip(test_set, test_set[1:]):
      loss += self.eval_model(Variable(np.array([cur_word], dtype=np.int32), volatile='on'), Variable(np.array([next_word], dtype=np.int32), volatile='on'))
    if loss is 0:
      return 0
    return np.exp(float(loss.data)/len(test_set))
