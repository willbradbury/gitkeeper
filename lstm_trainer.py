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

class RNN(Chain):
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
  def __init__(self, train, dev):
    self.rnn = RNN()
    self.model = L.Classifier(self.rnn)
    self.optimizer = optimizers.SGD()
    self.optimizer.setup(self.model)
    self.train_iter = ParallelSequentialIterator(train, 20)
    self.dev_iter = ParallelSequentialIterator(dev, 1, repeat=False)

  def get_trainer(self):
    updater = BPTTUpdater(self.train_iter, self.optimizer, 35, -1)
    trainer = training.Trainer(updater, (20, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(self.dev_iter, self.model))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch',
      'main/accuracy',
      'validation/main/accuracy']))
    return trainer
