import os
import codecs
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_file', default='')
parser.add_argument('--tgt_file', default='')
parser.add_argument('--mode', default='full', type=str, help='[part | full]')
parser.add_argument('--max_num', default=24*60, type=int)
args = parser.parse_args()


src_lst = []
with codecs.open(args.src_file, 'r', encoding='utf-8') as fin:
    for line in fin:
        if args.mode == 'part':
            if line.strip().split('\t')[-1] == '0':
                src_lst.append(line)
        else:
            src_lst.append(line)

index_lst = np.random.choice(len(src_lst), args.max_num, replace=False)

with codecs.open(args.tgt_file, 'w', encoding='utf-8') as fout:
    for i in range(len(src_lst)):
        if i in index_lst:
            fout.write(src_lst[i])