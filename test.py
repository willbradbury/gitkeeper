import sys
import util

verbosity = 3

def main():
  rp = util.download(sys.argv[1:][0], v=verbosity)
  rp.getDirList()

if __name__ == '__main__':
  main()