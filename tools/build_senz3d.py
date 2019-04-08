#! -*- coding: utf-8 -*-

import os
import codecs

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('--root_dir', default='./')
parser.add_argument('--subperson', default='P')
parser.add_argument('--num_person', default=10, type=int)
parser.add_argument('--subgesture', default='G')
parser.add_argument('--num_gesture', default=10, type=int)
parser.add_argument('--per_gesture', default=7, type=int)
parser.add_argument('--suffix', default='')
parser.add_argument('--save_path', default='./train.ntu.full.lst')

args = parser.parse_args()

def get_personal_pairs(opt):
  pair_lst = []
  for i in range(opt.num_gesture):
    for n in range(opt.per_gesture):
      for j in range(opt.num_gesture):
        for m in range(opt.per_gesture):
          if j!=i:
            pair_lst.append([i+1, n+1, j+1, m+1])
  return pair_lst

def build_dataset(opt):
  pair_lst = get_personal_pairs(opt)
  with codecs.open(opt.save_path, 'w', encoding='utf-8') as fout:
    for i in range(opt.num_person):
      for pl in pair_lst:
        im_A = os.path.join(opt.subperson+'%d'%(i+1),
                            opt.subgesture+'%d'%pl[0],
                            '%d'%pl[1]+opt.suffix)
        im_B = os.path.join(opt.subperson+'%d'%(i+1),
                            opt.subgesture+'%d'%pl[2],
                            '%d'%pl[3]+opt.suffix)
        fout.write('\t'.join([im_A, '%d'%pl[0], im_B, '%d'%pl[2]])+'\n')

if __name__ == '__main__':
  build_dataset(args)