import os.path
import random
import numpy as np
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms
from data.image_folder import make_gesture_dataset
from PIL import Image
import cv2

class GestureFullDataset(BaseDataset):
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
                                             transforms.Normalize((0.5,), (0.5,))])
        self.transform2 = transforms.Compose([transforms.ToTensor(),
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
        if len(AB_path) == 4:
            if random.random() > 0.5 and self.opt.phase == 'train':
                AB_path = [AB_path[2], AB_path[3], AB_path[0], AB_path[1]]

            A_path = os.path.join(self.data_dir, self.image_name, AB_path[0]+self.im_suffix)
            img_A = cv2.imread(A_path, cv2.IMREAD_UNCHANGED)
            img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
            cond_A = cv2.imread(os.path.join(self.data_dir, self.cond_type, AB_path[0]+self.im_suffix), 
                                cv2.IMREAD_GRAYSCALE)
            img_A = cv2.resize(img_A, (self.opt.load_size, self.opt.load_size), interpolation=cv2.INTER_CUBIC)
            cond_A = cv2.resize(cond_A, (self.opt.load_size, self.opt.load_size), interpolation=cv2.INTER_CUBIC)
            _, cond_A = cv2.threshold(cond_A, 64, 255, cv2.THRESH_BINARY)
            id_A = self._id2arr(int(AB_path[1])-1, self.opt.vdim)

            img_B = cv2.imread(os.path.join(self.data_dir, self.image_name, AB_path[2]+self.im_suffix), 
                               cv2.IMREAD_UNCHANGED)
            img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
            cond_B = cv2.imread(os.path.join(self.data_dir, self.cond_type, AB_path[2]+self.im_suffix), 
                                cv2.IMREAD_GRAYSCALE)
            img_B = cv2.resize(img_B, (self.opt.load_size, self.opt.load_size), interpolation=cv2.INTER_CUBIC)
            cond_B = cv2.resize(cond_B, (self.opt.load_size, self.opt.load_size), interpolation=cv2.INTER_CUBIC)
            _, cond_B = cv2.threshold(cond_B, 64, 255, cv2.THRESH_BINARY)
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
                img_B = self._im_trans(img_B, params)
                cond_B = self._im_trans(cond_B, params)
            else:
                img_A = Image.fromarray(img_A)
                cond_A = Image.fromarray(cond_A)
                img_B = Image.fromarray(img_B)
                cond_B = Image.fromarray(cond_B)

            # call standard transformation function
            img_A = self.transform2(img_A)
            img_B = self.transform2(img_B)
            cond_A = self.transform(cond_A)
            cond_B = self.transform(cond_B)

            if self.opt.phase == 'test':
                A_path = os.path.join(self.data_dir, self.image_name, ('-AB-'.join([AB_path[0],AB_path[2]])+self.im_suffix).replace('/', '-'))
            
            return {'A': img_A, 
                    'R_A': id_A,
                    'cond_A': cond_A,#self.cond_dict[AB_path[0]],
                    'B': img_B, 
                    'R_B': id_B,
                    'cond_B': cond_B,
                    'A_paths': A_path}#self.cond_dict[AB_path[2]]}
        else:
            A_path = os.path.join(self.data_dir, self.image_name, AB_path[0]+self.im_suffix)
            img_A = cv2.imread(A_path, cv2.IMREAD_UNCHANGED)
            img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
            img_A = cv2.resize(img_A, (self.opt.load_size, self.opt.load_size), interpolation=cv2.INTER_CUBIC)            
            cond_B = cv2.imread(os.path.join(self.data_dir, self.cond_type, AB_path[2]+self.im_suffix), 
                                cv2.IMREAD_GRAYSCALE)
            cond_B = cv2.resize(cond_B, (self.opt.load_size, self.opt.load_size), interpolation=cv2.INTER_CUBIC)
            
            if self.opt.geo_trans:
                cond_B = np.array(self._im_trans(cond_B, self._get_params(30,0.3,32,0)), dtype='uint8')

            _, cond_B = cv2.threshold(cond_B, 64, 255, cv2.THRESH_BINARY)
            img_A = self.transform2(Image.fromarray(img_A))
            cond_B = self.transform(Image.fromarray(cond_B))
            id_B = self._id2arr(int(AB_path[1])-1, self.opt.vdim)

            A_path = os.path.join(self.data_dir, self.image_name, '-'.join([AB_path[0], AB_path[2], AB_path[1]]).replace('/','-'))
            if self.opt.geo_trans:
                A_path += str(index) + self.im_suffix
            else:
                A_path += self.im_suffix

            return {'A': img_A, 'R_B': id_B, 'cond_B': cond_B, 'A_paths': A_path}            

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

