""" Classes to handle local (Repo) and remote (RemoteRepo) git repositories.
    (c) Alex Wang, Shivaal Roy, Will Bradbury
"""
import os
import util

default_repo_root = os.environ['HOME']+"/gitkeeper/repos/"

class Repo(object):
  def __init__(self, name, root=default_repo_root):
    self.name = name
    self.id = self.name.replace('/', '.')
    self.exists = True
    if not os.path.exists(default_repo_root + self.id):
        self.exists = False

class RemoteRepo(object):
  def __init__(self, name):
    self.name = name
    self.id = self.name.replace('/', '.')

  def download(self):
    """download from github"""
    util.createAndCd(default_repo_root)
    util.createAndCd(self.id)
    util.createAndCd('train')
    # download train data
    util.downloadPullRequests(self.name, 40000, 40001)
    util.go_to_parent()
    util.createAndCd('test')
    # download test data
    #util.downloadPullRequests(self.name, 18381,)
    util.go_to_parent()
    os.system("git clone git@github.com:"+ self.name + ".git")
    return self

  def getRepo(self):
    """turn the remote repo into a local repo"""
    return Repo(self.name)