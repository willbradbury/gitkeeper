""" Helpful utility functions for file and github tasks
    (c) Alex Wang, Shivaal Roy, Will Bradbury
"""
import urllib2, os, json
from repo import Repo, RemoteRepo

github_access_token = os.environ['GITHUBTOKEN']

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
      # attempt to download metadata and diff information
      metadata = urllib2.urlopen(
         'http://api.github.com/repos/' + repo_name + '/pulls/' + str(diff) + \
             "?access_token=" + github_access_token)
      if not metadata:
        log(v,3,str(diff) + " does not have metadata: " + repo_name)
        continue

      diff_text = urllib2.urlopen(
         'http://www.github.com/' + repo_name + '/pull/' + str(diff) + '.diff')
      if "pull" not in diff_text.url:
        log(v,3,str(diff) + " is not a pull request in " + repo_name)
        continue

      # write to files after downloading
      open(str(diff) + '.metadata', 'wrb').write(metadata.read())
      open(str(diff) + '.diff', 'wrb').write(diff_text.read())
      log(v,3,"successfully downloaded pull request " + str(diff) + " from "+ repo_name)
    except Exception as e:
      # there was an error fetching the pull request.
      # most likely, the pull request doesn't exist at all.
      log(v,4,str(e))

def go_to_parent():
  os.chdir('../')

def log(verbosity, level, message):
  """ Log a message if its log level is at or above the verbosity
      set in the parameters."""
  if verbosity >= level:
    print message
