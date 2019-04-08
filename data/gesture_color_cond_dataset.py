import os.path
import random
import numpy as np
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms
from data.image_folder import make_gesture_dataset
from PIL import Image
import cv2

class GestureColorCondDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        #self.opt = opt
        self.im_suffix = '.png'
        self.image_name = opt.image_name
        self.cond_type = opt.cond_type
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

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            R (tensor) - - its corresponding landuse vector
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image and its corresponding label map given a random integer index

        AB_path = self.AB_paths[index]
        if self.opt.phase == 'train':
            if random.random() > 0.5:
                AB_path = [AB_path[2], AB_path[3], AB_path[0], AB_path[1]]

            img_A = cv2.imread(os.path.join(self.data_dir, self.image_name, AB_path[0]+self.im_suffix), 
                               cv2.IMREAD_UNCHANGED)
            img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
            cond_A = cv2.imread(os.path.join(self.data_dir, self.cond_type, AB_path[0]+self.im_suffix), 
                                cv2.IMREAD_UNCHANGED)
            cond_A = cv2.cvtColor(cond_A, cv2.COLOR_BGR2RGB)
            img_A = cv2.resize(img_A, (self.opt.load_size, self.opt.load_size), interpolation=cv2.INTER_CUBIC)
            cond_A = cv2.resize(cond_A, (self.opt.load_size, self.opt.load_size), interpolation=cv2.INTER_CUBIC)
            id_A = self._id2arr(int(AB_path[1])-1, self.opt.vdim)

            img_B = cv2.imread(os.path.join(self.data_dir, self.image_name, AB_path[2]+self.im_suffix), 
                               cv2.IMREAD_UNCHANGED)
            img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
            cond_B = cv2.imread(os.path.join(self.data_dir, self.cond_type, AB_path[2]+self.im_suffix), 
                                cv2.IMREAD_UNCHANGED)
            cond_B = cv2.cvtColor(cond_B, cv2.COLOR_BGR2RGB)
            img_B = cv2.resize(img_B, (self.opt.load_size, self.opt.load_size), interpolation=cv2.INTER_CUBIC)
            cond_B = cv2.resize(cond_B, (self.opt.load_size, self.opt.load_size), interpolation=cv2.INTER_CUBIC)
            id_B = self._id2arr(int(AB_path[3])-1, self.opt.vdim)

            # apply the same flipping to both A and B
            if (not self.opt.no_flip) and random.random() > 0.5:
                img_A = np.fliplr(img_A)
                cond_A = np.fliplr(cond_A)
                img_B = np.fliplr(img_B)
                cond_B = np.fliplr(cond_B)

            if self.opt.geo_trans:
                params = self._get_params()
                img_A = self._im_trans(img_A, params)
                cond_A = self._im_trans(cond_A, params)
                #params = self._get_params()
                img_B = self._im_trans(img_B, params)
                cond_B = self._im_trans(cond_B, params)
            else:
                img_A = Image.fromarray(img_A)
                cond_A = Image.fromarray(cond_A)
                img_B = Image.fromarray(img_B)
                cond_B = Image.fromarray(cond_B)
            #print(params, img_A.size, cond_A.size)

            # call standard transformation function
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
            cond_A = self.transform(cond_A)
            cond_B = self.transform(cond_B)

            # exchange the value of A and B
            return {'A': img_A, 
                    'R_A': id_A,
                    'cond_A': cond_A,#self.cond_dict[AB_path[0]],
                    'B': img_B, 
                    'R_B': id_B,
                    'cond_B': cond_B}#self.cond_dict[AB_path[2]]}
        else:
            A_path = os.path.join(self.data_dir, self.image_name, AB_path[0]+self.im_suffix)
            img_A = cv2.imread(A_path, cv2.IMREAD_UNCHANGED)
            img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
            img_A = cv2.resize(img_A, (self.opt.load_size, self.opt.load_size), interpolation=cv2.INTER_CUBIC)            
            cond_B = cv2.imread(os.path.join(self.data_dir, self.cond_type, AB_path[2]+self.im_suffix), 
                                cv2.IMREAD_UNCHANGED)
            cond_B = cv2.cvtColor(cond_B, cv2.COLOR_BGR2RGB)
            cond_B = cv2.resize(cond_B, (self.opt.load_size, self.opt.load_size), interpolation=cv2.INTER_CUBIC)
            #_, cond_B = cv2.threshold(cond_B, 127, 255, cv2.THRESH_BINARY)
            img_A = self.transform(Image.fromarray(img_A))
            cond_B = self.transform(Image.fromarray(cond_B))
            id_B = self._id2arr(int(AB_path[1])-1, self.opt.vdim)

            return {'A': img_A, 'R_B': id_B, 'cond_B': cond_B, 'A_paths': A_path}            

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

