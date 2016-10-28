""" Helpful utility functions for file and github tasks
    (c) Alex Wang, Shivaal Roy, Will Bradbury
"""
from repo import Repo, RemoteRepo

def download(repo_name):
  """ Try to open |repo_name| locally, but if it doesn't exist,
      download it from github/repo_name."""
  rp = Repo(name=repo_name)
  if not rp.exists:
      return RemoteRepo(name=repo_name).download().getRepo()
  else:
      return rp
  print rp.name
