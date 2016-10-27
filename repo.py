""" Classes to handle local (Repo) and remote (RemoteRepo) git repositories.
    (c) Alex Wang, Shivaal Roy, Will Bradbury
"""
import os

default_repo_root = os.environ['HOME']+"/gitkeeper/repos/"

class Repo(object):
  def __init__(self, name, root=default_repo_root):
    self.name = name
    pass
  
class RemoteRepo(object):
  def __init__(self, name):
    self.name = name
    pass

  def download(self):
    """download from github"""
    return self

  def getRepo(self):
    """turn the remote repo into a local repo"""
    return Repo(self.name)
