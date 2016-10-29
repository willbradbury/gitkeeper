""" Helpful utility functions for file and github tasks
    (c) Alex Wang, Shivaal Roy, Will Bradbury
"""
from repo import Repo, RemoteRepo
import urllib2
import os
import json

def download(repo_name):
  """ Try to open |repo_name| locally, but if it doesn't exist,
      download it from github/repo_name."""
  rp = Repo(name=repo_name)
  if not rp.exists:
      return RemoteRepo(name=repo_name).download().getRepo()
  else:
      return rp

def createAndCd(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.chdir(directory)

def downloadPullRequests(repo_name, start_index, end_index):
  for diff in xrange(start_index, end_index):
    # urllib2.urlretrieve('http://api.github.com/repos/' + repo_name + '/pulls/' + str(diff), str(diff) + '.metadata')
    # urllib2.request.retrieve()
    open(str(diff) + '.metadata', 'wrb').write(urllib2.urlopen(
       'http://api.github.com/repos/' + repo_name + '/pulls/' + str(diff)).read())
    open(str(diff) + '.diff', 'wrb').write(urllib2.urlopen(
       'http://www.github.com/' + repo_name + '/pull/' + str(diff) + '.diff').read())
    print 'http://api.github.com/repos/' + repo_name + '/pulls/' + str(diff)
    print 'http://www.github.com/' + repo_name + '/pull/' + str(diff) + '.diff'

def go_to_parent():
  os.chdir('../')