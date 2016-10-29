""" Helpful utility functions for file and github tasks
    (c) Alex Wang, Shivaal Roy, Will Bradbury
"""
from repo import Repo, RemoteRepo
import urllib2
import os
import json

def download(repo_name, v=1):
  """ Try to open |repo_name| locally, but if it doesn't exist,
      download it from github/repo_name."""
  rp = Repo(name=repo_name, v=v)
  if not rp.exists:
      return RemoteRepo(name=repo_name, v=v).download().getRepo()
  else:
      return rp

def createAndCd(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.chdir(directory)

def downloadPullRequests(repo_name, start_index, end_index, v=1):
  for diff in xrange(start_index, end_index):
    try:
      open(str(diff) + '.metadata', 'wrb').write(urllib2.urlopen(
         'http://api.github.com/repos/' + repo_name + '/pulls/' + str(diff)).read())
      open(str(diff) + '.diff', 'wrb').write(urllib2.urlopen(
         'http://www.github.com/' + repo_name + '/pull/' + str(diff) + '.diff').read())
      log(v,3,"successfully downloaded pull request " + str(diff) + " from "+ repo_name)
    except Exception as e:
      log(v,3,str(e))

def go_to_parent():
  os.chdir('../')

def log(verbosity, level, message):
  if verbosity >= level:
    print message
