"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import codecs
from collections import OrderedDict
import random
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_data_dir(opt):
    return os.path.join()


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def make_gesture_part_dataset(img_path, 
                              im_list, 
                              max_dataset_size=float('inf'),
                              phase='train'):
    images = []
    assert os.path.isdir(img_path), '%s is not a valid directory' % img_path
    assert os.path.isfile(im_list), '%s is not a valid file' % im_list
    with codecs.open(im_list, 'r', encoding='utf-8') as fin:
        for line in fin:
            items = line.strip().split('\t')
            if len(items) == 5:
                img_A, id_A, img_B, id_B, flag = items
                if is_image_file(os.path.join(img_path, img_A+'.png')) and \
                    is_image_file(os.path.join(img_path, img_B+'.png')):
                    images.append([img_A, id_A, img_B, id_B, flag])
            elif phase == 'test' and len(items)==3:
                img_A, id_B, cond_B = items
                if is_image_file(os.path.join(img_path, img_A+'.png')) and \
                    is_image_file(os.path.join(img_path, cond_B+'.png')):
                    images.append([img_A, id_B, cond_B])
            else:
                print('Unknown load mode.')
    return images[:min(max_dataset_size, len(images))]


def make_gesture_dataset(img_path, 
                         im_list, 
                         max_dataset_size=float('inf'), 
                         phase='train'):
    images = []
    assert os.path.isdir(img_path), '%s is not a valid directory' % img_path
    assert os.path.isfile(im_list), '%s is not a valid file' % im_list
    with codecs.open(im_list, 'r', encoding='utf-8') as fin:
        for line in fin:
            items = line.strip().split('\t')
            if len(items) == 4:
                img_A, id_A, img_B, id_B = items
                if is_image_file(os.path.join(img_path, img_A+'.png')) and \
                    is_image_file(os.path.join(img_path, img_B+'.png')):
                    images.append([img_A, id_A, img_B, id_B])
            elif phase == 'test' and len(items) == 3:
                img_A, id_B, cond_B = items
                if is_image_file(os.path.join(img_path, img_A+'.png')) and \
                    is_image_file(os.path.join(img_path, cond_B+'.png')):
                    images.append([img_A, id_B, cond_B])
            else:
                print('Unknown load mode.')
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)

def imread(path, load_size, load_mode=cv2.IMREAD_UNCHANGED, convert_rgb=False, thresh=-1):
    im = cv2.imread(path, load_mode)
    if convert_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    if thresh > 0:
        _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    return im
