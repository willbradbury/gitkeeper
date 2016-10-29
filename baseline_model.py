""" Baseline model implementation.
    (c) Alex Wang, Shivaal Roy, Will Bradbury"""

import numpy as np
from sklearn.svm import SVC
import json
import util
import random

class BaselineModel(object):
  def __init__(self, repo, v=1):
    self.v = v
    self.extractor = BaselineExtractor()
    self.repo = repo
    self.clf = SVC()

  def train(self):
    X,y = np.array([]), np.array([])

    util.log(self.v, 1, "beginning training on " + self.repo.name)
    for i,pid in enumerate(self.repo.getExamples(inTraining=True)):
      # grab the files for this training example
      diff_f = self.repo.getDiffFile(pid, inTraining=True)
      meta_f = self.repo.getMetaFile(pid, inTraining=True)

      meta_json = json.load(meta_f)
      features = self.extractor.extract(diff_f, meta_json)
      X = np.reshape(X, (-1, features.size))
      if i is 0:
        X = np.append(X, features)
      else:
        X = np.append(X,features,axis=0)
      y = np.append(y,self.extractor.label(meta_json), axis=0)

    self.clf.fit(X,y)
    score = self.clf.score(X,y)
    util.log(self.v, 1, "mean training accuracy: " + str(score))

  def test(self):
    X,y = np.array([]), np.array([])

    util.log(self.v, 1, "beginning testing on " + self.repo.name)
    for i,pid in enumerate(self.repo.getExamples(inTraining=False)):
      # grab the files for testing
      diff_f = self.repo.getDiffFile(pid, inTraining=False)
      meta_f = self.repo.getMetaFile(pid, inTraining=False)

      meta_json = json.load(meta_f)
      features = self.extractor.extract(diff_f, meta_json)
      X = np.reshape(X, (-1, features.size))
      if i is 0:
        X = np.append(X, features)
      else:
        X = np.append(X,features,axis=0)
      y = np.append(y,self.extractor.label(meta_json), axis=0)

    score = self.clf.score(X,y)
    util.log(self.v, 1, "mean testing accuracy: " + str(score))

class BaselineExtractor(object):
  def __init__(self, v=1):
    self.v = v

  def extract(self, diff_f, meta_json):
    return np.array([[1]])

  def label(self, meta_json):
    if meta_json['merged']: return 1
    return -1
