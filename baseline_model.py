""" Baseline model implementation.
    (c) Alex Wang, Shivaal Roy, Will Bradbury"""

import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import json, util, random
import model

class BaselineModel(model.Model):
  def __init__(self, repo, v=1):
    self.v = v
    self.extractor = BaselineExtractor()
    self.repo = repo
    self.clf = SVC()

  def train(self):
    y = np.array([])
    corpus = []

    util.log(self.v, 1, "beginning training on " + self.repo.name)
    for i,pid in enumerate(self.repo.getExamples(inTraining=True)):
      # grab the files for this training example
      diff_f = self.repo.getDiffFile(pid, inTraining=True)
      meta_f = self.repo.getMetaFile(pid, inTraining=True)

      meta_json = json.load(meta_f)
      corpus.append(diff_f.read())
      y = np.append(y,self.extractor.label(meta_json))

    self.vectorizer = CountVectorizer(min_df=1)
    X = self.vectorizer.fit_transform(corpus)
    self.clf.fit(X,y)
    score = self.clf.score(X,y)
    util.log(self.v, 1, "mean training accuracy: " + str(score))

  def test(self):
    y = np.array([])
    test_diffs = []

    util.log(self.v, 1, "beginning testing on " + self.repo.name)
    for i,pid in enumerate(self.repo.getExamples(inTraining=False)):
      # grab the files for testing
      diff_f = self.repo.getDiffFile(pid, inTraining=False)
      meta_f = self.repo.getMetaFile(pid, inTraining=False)

      meta_json = json.load(meta_f)
      test_diffs.append(diff_f.read())
      y = np.append(y,self.extractor.label(meta_json))

    X = self.vectorizer.transform(test_diffs)
    score = self.clf.score(X,y)
    util.log(self.v, 1, "mean testing accuracy: " + str(score))

class BaselineExtractor(object):
  def __init__(self, v=1):
    self.v = v

  def extract(self, diff_f, meta_json):
    return None

  def label(self, meta_json):
    if meta_json['merged']: return 1
    return -1
