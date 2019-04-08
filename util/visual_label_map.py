# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--im_path')
parser.add_argument('--save_path')
parser.add_argument('--num_cls', default=13, type=int, help='amount of categories')
args = parser.parse_args()

def uint82bin(n, count=8):
  """returns the binary of integer n, count refers to amount of bits"""
  return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def label_colormap(N=13):
  cmap = np.zeros((N, 3), dtype = np.uint8)
  for i in range(N):
    r = 0
    g = 0
    b = 0
    id = i
    for j in range(7):
      str_id = uint82bin(id)
      r = r ^ ( np.uint8(str_id[-1]) << (7-j))
      g = g ^ ( np.uint8(str_id[-2]) << (7-j))
      b = b ^ ( np.uint8(str_id[-3]) << (7-j))
      id = id >> 3
    cmap[i, 0] = r
    cmap[i, 1] = g
    cmap[i, 2] = b
  return cmap

def label_visualizer(img, color_map=None):
  if color_map is None:
    return img

  m,n = img.shape
  color_img = np.zeros((m,n,3), dtype=np.uint8)
  for i in range(m):
    for j in range(n):
      color_img[i,j,:] = np.array([color_map[img[i,j],:]])
  return color_img

def image_save(img, save_path):
  image = Image.fromarray(img, 'RGB')
  image.save(save_path, format='png')

if __name__ == '__main__':
  image = np.array(Image.open(args.im_path))
  color_map = label_colormap(args.num_cls)
  color_image = label_visualizer(image, color_map)
  image_save(color_image, args.save_path)
