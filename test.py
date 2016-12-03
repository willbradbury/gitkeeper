import sys
import util
import re

verbosity = 3

def main():
  l =  re.split(r'(\.|,|\\|/|#|!|\$|%|\^|&|\*|;|:|{|}|=|-|_|`|~|\(|\)| |\t)', 'aa.bb,cc\dd/ee#ff!gg$hh%ii\jj^kk&ll\mm*nn;oo:         \t    \t\tpp{qq}rr=ss\tt-uu_vv`ww~xx(yy)zz')
  q=[x for x in l if x != '']
  print q
  print [x for x,y in [(z1,z2) for z1,z2 in zip(q,q[1:]) if not(z1==z2 and z1 in [' ', '\t', '\n'])]]
if __name__ == '__main__':
  main()