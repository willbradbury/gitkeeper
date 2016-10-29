""" Classes to handle local (Repo) and remote (RemoteRepo) git repositories.
    (c) Alex Wang, Shivaal Roy, Will Bradbury
"""
import os
import util
from os.path import isfile, join

default_repo_root = os.environ['HOME']+"/gitkeeper/repos/"

class Repo(object):
  def __init__(self, name, root=default_repo_root, v=1):
    self.name = name
    self.id = self.name.replace('/', '.')
    self.v = v
    if not os.path.exists(default_repo_root + self.id):
      util.log(self.v, 2, "cannot find repo locally: " + self.name)
      self.exists = False
    else:
      util.log(self.v, 2, "found repo locally: " + self.name)
      self.exists = True

  def getExamples(self, inTraining):
    folder_path = default_repo_root + self.id + ("/train" if inTraining else "/test")
    example_files = [int(f.split('.')[0]) \
        for f in os.listdir(folder_path) if isfile(join(folder_path, f)) and f[0] is not '.']
    pull_ids = sorted(list(set(example_files)))
    util.log(self.v, 3, "listing examples for " + self.name)
    for pid in pull_ids:
      yield pid

  def getDiffFile(self, pid, inTraining):
    folder = default_repo_root + self.id + ("/train/" if inTraining else "/test/")
    return open(folder + str(pid) + ".diff", 'rb')
  
  def getMetaFile(self, pid, inTraining):
    folder = default_repo_root + self.id + ("/train/" if inTraining else "/test/")
    return open(folder + str(pid) + ".metadata", 'rb')

class RemoteRepo(object):
  def __init__(self, name, v=1):
    self.name = name
    self.id = self.name.replace('/', '.')
    self.v = v

  def download(self):
    """download from github"""
    util.createAndCd(default_repo_root)
    util.createAndCd(self.id)
    util.createAndCd('train')
    # download train data
    util.downloadPullRequests(self.name, 1, 4384, v=self.v)
    util.go_to_parent()
    util.createAndCd('test')
    # download test data
    util.downloadPullRequests(self.name, 4384, 5479, v=self.v)
    util.go_to_parent()
    os.system("git clone git@github.com:"+ self.name + ".git")
    util.log(self.v, 2, "downloaded repo " + self.name)
    return self

  def getRepo(self):
    """turn the remote repo into a local repo"""
    return Repo(self.name)
