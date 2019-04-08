#! -*- coding: utf-8 -*-

import os
import cv2
import glob
import numpy as np
import codecs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='')
parser.add_argument('--save_dir', default='')
parser.add_argument('--output_file', default='')
parser.add_argument('--anno_type', default='')
args = parser.parse_args()

def is_pairs(data_dir, fname, suffix):
  flag_A = os.path.isfile(os.path.join(data_dir, fname))
  flag_B = os.path.isfile(os.path.join(data_dir, fname.replace(suffix, '')))
  return flag_A and flag_B

def get_img_list(data_dir, suffix='.png'):
  im_list = glob.glob(os.path.join(data_dir, '*'+suffix))
  new_im_list = []
  for im in im_list:
    fname = im.split('/')[-1]
    #if is_pairs(data_dir, fname, suffix.split('.')[0]):
    new_im_list.append(fname)
  return new_im_list

def build_dataset_dict(im_list):
  dataset_dict = {}
  for im in im_list:
    key = parsing_fname(im)[0]
    if key not in dataset_dict:
      dataset_dict[key] = 0
  return dataset_dict

def parsing_fname(fname, seperator='_color_'):
  fname = fname.split('.')[0].strip('color_AB')
  return fname.split(seperator)

def parsing_dir(fname, suffix='color.png'):
  person_id, gesture_id, index = fname.split('_')
  #name = index + suffix
  return person_id, gesture_id, index

def create_dir(data_root, person_id, gesture_id, subdata_root=''):
  if not os.path.exists(os.path.join(data_root, subdata_root, person_id)):
    os.mkdir(os.path.join(data_root, subdata_root, person_id))
  if not os.path.exists(os.path.join(data_root, subdata_root, person_id, gesture_id)):
  	os.mkdir(os.path.join(data_root, subdata_root, person_id, gesture_id))

def img_read(fname, raw_type=True, use_rgb=True, 
             use_binary=False, binary_params=(127, 1), 
             resize=None):
  assert(os.path.isfile(fname)), '{} is not a existing file.'.format(fname) 
  
  flag = cv2.IMREAD_UNCHANGED if raw_type else cv2.IMREAD_GRAYSCALE
  im = cv2.imread(fname, flag)
  if resize is not None:
    im = cv2.resize(im, resize, interpolation=cv2.INTER_CUBIC)
  if use_rgb:
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  if use_binary:
    _, im = cv2.threshold(im, binary_params[0], binary_params[1], cv2.THRESH_BINARY)
  return im

def img_split(im, num_col=3):
  h, w, _ = im.shape
  w2 = w//num_col
  return im[:,:w2,:], im[:,w2:w2*2,:], im[:,w2*2:,:]

def img_binary(im, params=(64, 255), use_rgb=False):
  if len(im.shape) > 2:
    flag = cv2.COLOR_RGB2GRAY if use_rgb else cv2.COLOR_BGR2GRAY
    im = cv2.cvtColor(im, flag)
  _, im = cv2.threshold(im, params[0], params[1], cv2.THRESH_BINARY)
  return im

def img_save(save_path, fname, im):
  assert(os.path.isdir(save_path)), save_path
  flags = [cv2.IMWRITE_PNG_COMPRESSION, 0]
  cv2.imwrite(os.path.join(save_path, fname), im, flags)

def write2file(out_file, out_path):
  with codecs.open(out_path, 'w', encoding='utf-8') as fout:
    for oo in out_file:
      fout.write(oo+'\n')

def trans_dataset(data_dir, save_dir, output_file, anno_type='keypoint'):
  img_list = get_img_list(data_dir)
  dataset_dict = build_dataset_dict(img_list)
  out_list = []
  for im in img_list:
    name_A, name_B = parsing_fname(im)
    out_list.append('\t'.join([name_A, name_B]))
    if name_B in dataset_dict:
      if dataset_dict[name_B] == 0:
        cur_img = img_read(os.path.join(data_dir, im), use_rgb=True)
        _, B, cond_B = img_split(cur_img)

        p_id, g_id, idx = parsing_dir(name_B)
        #print(p_id, g_id, idx)
        create_dir(save_dir, p_id, g_id, 'images')
        cur_img_save_path = os.path.join(save_dir, 'images', p_id, g_id)
        create_dir(save_dir, p_id, g_id, anno_type)
        cur_lab_save_path = os.path.join(save_dir, anno_type, p_id, g_id)
        img_save(cur_img_save_path, idx+'-color.png', B)
        cond_B = img_binary(cond_B, use_rgb=True)
        img_save(cur_lab_save_path, idx+'-color.png', cond_B)

        dataset_dict[name_B] = 1
  write2file(out_list, output_file)

if __name__ == '__main__':
  '''
  data_dir = './1_ntu_point_black_good/train'
  save_dir = './ntu_dataset'
  output_file = './ntu_dataset/train.lst'
  anno_type = 'keypoint'
  '''
  trans_dataset(args.data_dir, 
                args.save_dir, 
                args.output_file, 
                args.anno_type)
