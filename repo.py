""" Classes to handle local (Repo) and remote (RemoteRepo) git repositories.
    (c) Alex Wang, Shivaal Roy, Will Bradbury
"""
import os

default_repo_root = os.environ['HOME']+"/gitkeeper/repos/"

class Repo(object):
  def __init__(self, name, root=default_repo_root):
    self.name = name
    self.repoName = self.name.split('/')[-1]
    self.exists = True
    if not os.path.exists(default_repo_root + self.repoName):
        self.exists = False

class RemoteRepo(object):
  def __init__(self, name):
    self.name = name

  def download(self):
    """download from github"""
    if not os.path.exists(default_repo_root):
        os.makedirs(default_repo_root)
    os.chdir(default_repo_root)
    os.system("git clone git@github.com:"+ self.name + ".git")
    return self

  def getRepo(self):
    """turn the remote repo into a local repo"""
    return Repo(self.name)
