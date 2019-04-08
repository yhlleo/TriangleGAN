#! -*- coding: utf-8 -*-

'''
  Usage: python3 eval.py --metric_mode mse --model_name ntu_part_gesturegan_skeleton --suffix_pred fake_B
'''

import os
import numpy as np
import data_io
import ntpath
import json
from scipy.misc import imread
from basic_scores import mse, psnr
from inception_score import inception_score
from fid_scores import cal_fid as fid_score
from frd_scores import cal_frechet_resnet_distance as frd_score
from prd_score import prd_score

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--metric_mode', type=str, default='mse', help='[mse | psnr | is | fid | frd | prd]')
parser.add_argument('--model_name', type=str, default='ntu_part_gesturegan_skeleton')
parser.add_argument('--results_dir', type=str, default='../results')
parser.add_argument('--suffix_gt', type=str, default='real_B')
parser.add_argument('--suffix_pred', type=str, default='fake_B', help='[fake_B | fake_B_masked | fake_B2_masked]')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--gpu_id', type=str, default='')
parser.add_argument('--output_json', type=str, default='')
args = parser.parse_args()

def print_eval_log(opt):
    message = ''
    message += '----------------- Eval ------------------\n'
    for k, v in sorted(opt.items()):
        message += '{:>20}: {:<10}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)


if __name__ == '__main__':
    use_cuda = args.gpu_id != ''
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    batch_size = args.batch_size
    metric_mode = args.metric_mode

    results_dir = os.path.join(args.results_dir, args.model_name, 'test_latest', 'images')
    src_img_list, tgt_img_list = data_io.get_image_pairs(results_dir, args.suffix_gt, args.suffix_pred)

    final_score = 0.0
    if metric_mode == 'mse' or metric_mode == 'psnr':
        scores = []
        for src, tgt in zip(src_img_list, tgt_img_list):
            im_src = imread(src).astype(np.float32)
            im_tgt = imread(tgt).astype(np.float32)

            if metric_mode == 'mse':
                scores.append(mse(im_src, im_tgt))
            else:
                scores.append(psnr(im_src, im_tgt))
        final_score = np.mean(scores)
    elif metric_mode == 'is':
        data_generator = data_io.data_prepare(tgt_img_list, batch_size, use_cuda)
        final_score = inception_score(data_generator, use_cuda)
    elif metric_mode == 'fid':
        src_data_generator = data_io.data_prepare(src_img_list, batch_size, use_cuda)
        tgt_data_generator = data_io.data_prepare(tgt_img_list, batch_size, use_cuda)
        dims = 2048
        final_score = fid_score(src_data_generator, tgt_data_generator, dims, use_cuda)
    elif metric_mode == 'frd': # TODO
        src_data_generator = data_io.data_prepare(src_img_list, batch_size, use_cuda)
        tgt_data_generator = data_io.data_prepare(tgt_img_list, batch_size, use_cuda)
        dims = 1000
        final_score = frd_score(src_data_generator, tgt_data_generator, dims, use_cuda)
    elif metric_mode == 'prd':
        src_data_generator = data_io.data_prepare(src_img_list, batch_size, use_cuda)
        tgt_data_generator = data_io.data_prepare(tgt_img_list, batch_size, use_cuda)
        dims = 1000
        final_score = prd_score(src_data_generator, tgt_data_generator, dims, use_cuda)
        score = {"recall": final_score[0].tolist(),
                 "precision": final_score[1].tolist()}
        result = {"label": args.model_name,
                  "score": score}
        if args.output_json == '':
            args.output_json = '{}.json'.format(args.model_name)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, 
                      ensure_ascii=False, indent=4, 
                      sort_keys=True, separators=(',', ': '))
        final_score = 1.0
    else:
        print('Unknown metric mode.')

    logs = {'model_name': args.model_name,
            'num_of_files': len(tgt_img_list),
            'metric_mode': metric_mode,
            'final_score': final_score}
    print_eval_log(logs)