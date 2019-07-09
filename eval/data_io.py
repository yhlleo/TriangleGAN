import os
import numpy as np
import cv2
import torch
import glob
import torch.nn.functional as F
import torchvision.transforms as transforms

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

def imread(file):
    im = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# build an iterable generator 
def data_prepare(files, batch_size=8, use_cuda=False):
    assert len(files), len(files)
    n_batches = len(files) // batch_size

    if len(files) % batch_size != 0:
        n_batches += 1

    if batch_size > len(files):
        batch_size = len(files)

    for i in tqdm(range(n_batches)):
        #print('\rPropagating batch %d/%d' % (i+1, n_batches))
        start = i * batch_size
        end = start + batch_size
        end = end if end <= len(files) else len(files)
        
        images = np.array([imread(str(f)).astype(np.float32)
                       	   for f in files[start:end]])

        images = images.transpose((0, 3, 1, 2))
        images /= 255.0

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if use_cuda:
            batch = batch.cuda()
        yield batch

def get_image_pairs(data_dir, suffix_gt='real_B', suffix_pred='fake_B'):
    gt_list = glob.glob(os.path.join(data_dir, '*{}.png'.format(suffix_gt)))
    pred_list = [ll.replace(suffix_gt, suffix_pred) for ll in gt_list]
    return gt_list, pred_list
