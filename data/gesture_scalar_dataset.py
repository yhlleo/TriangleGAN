# Author: Yahui Liu <yahui.liu@unitn.it>

import os.path
import random
import numpy as np
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms
from data.image_folder import make_gesture_dataset
from PIL import Image

class CondGestureV1Dataset(BaseDataset):
    """A dataset class for paired image dataset."""

    def __init__(self, opt):
        """Initialize this dataset class."""
        BaseDataset.__init__(self, opt)
        self.im_suffix = '.png'
        self.image_name = opt.image_name
        if self.opt.phase == 'train':
            self.data_dir = os.path.join(self.opt.dataroot,
                                         self.opt.data_name)
            self.AB_paths = make_gesture_dataset(os.path.join(self.data_dir, self.opt.image_name), 
                                                 os.path.join(self.data_dir, self.opt.train_name), 
                                                 self.opt.max_dataset_size,
                                                 self.opt.phase)
            random.seed(1234)
            random.shuffle(self.AB_paths)
            assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        else:
            self.data_dir = os.path.join(self.opt.dataroot,
                                         self.opt.data_name)
            self.AB_paths = make_gesture_dataset(os.path.join(self.data_dir, self.opt.image_name),
                                                 os.path.join(self.data_dir, self.opt.test_name), 
                                                 self.opt.max_dataset_size,
                                                 self.opt.phase)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5),
                                                                  (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """
        # read a image and its corresponding label map given a random integer index
        AB_path = self.AB_paths[index]
        if self.opt.phase == 'train':
            if random.random() > 0.5:
                AB_path = [AB_path[2], AB_path[3], AB_path[0], AB_path[1]]

            img_A = Image.open(os.path.join(self.data_dir, self.image_name, AB_path[0]+self.im_suffix)).convert('RGB')
            img_A = img_A.resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)
            id_A = self._id2arr(int(AB_path[1])-1, self.opt.vdim)

            img_B = Image.open(os.path.join(self.data_dir, self.image_name, AB_path[2]+self.im_suffix)).convert('RGB')
            img_B = img_B.resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)
            id_B = self._id2arr(int(AB_path[3])-1, self.opt.vdim)

            # apply the same flipping to both A and B
            if (not self.opt.no_flip) and random.random() > 0.5:
                img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
                img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)

            # call standard transformation function
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
            # exchange the value of A and B
            return {'A': img_A, 
                    'R_A': id_A,
                    'B': img_B, 
                    'R_B': id_B}
        else:
            A_path = os.path.join(self.data_dir, self.image_name, AB_path[0]+self.im_suffix)
            img_A = Image.open(A_path).convert('RGB')
            img_A = img_A.resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)
            img_A = self.transform(img_A)
            id_B = self._id2arr(int(AB_path[1])-1, self.opt.vdim)

            return {'A': img_A, 'R_B': id_B, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

