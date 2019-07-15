import torch
import itertools
import torch.nn.functional as F
import numpy as np
from util.image_pool import ImagePool
from .base_model import BaseModel
from torchvision.models import vgg16

from . import networks
from . import gesturegan_networks as gesturegan

class GestureGANRawModel(BaseModel):
    """
    GestureGAN
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_color', type=float, default=100, help='weight for color loss')
            parser.add_argument('--lambda_cycle', type=float, default=10.0, help='weight for cycle reconstruction')
            parser.add_argument('--lambda_rec', type=float, default=1.0, help='weight for reconstruction loss: G(A, S_B) -> B, G(B, S_B) -> A')
            parser.add_argument('--lambda_adv', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--lambda_idt', type=float, default=0.1, help='use identity mapping.')
            
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D', 'GAB', 'rec', 'cycle', 'idt', 'color']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        #self.visual_names = ['real_A', 'cond_B', 'fake_B', ]  # combine visualizations for A
        self.visual_names = ['real_B', 'fake_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # define networks (both Generators and discriminators)
        self.netG = gesturegan.define_GGesture(opt.input_nc+opt.cond_dim, # (3+1)x256x256
                                               opt.output_nc,
                                               opt.ngf,
                                               opt.norm,
                                               not opt.no_dropout,
                                               opt.init_type,
                                               opt.init_gain,
                                               self.gpu_ids)  # out_nc: 3x256x256

        if self.isTrain:  # define discriminators
            self.netD = gesturegan.define_DGesture(opt.output_nc, 
                                                   opt.ndf,
                                                   opt.n_layers_D,
                                                   opt.norm, 
                                                   opt.init_type, 
                                                   opt.init_gain, 
                                                   self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionRec1 = torch.nn.L1Loss()
            self.criterionRec2 = torch.nn.MSELoss()

            self.vgg_mdoel = vgg16(pretrained=True).to(self.device).cuda()
            self._set_parameter_requires_grad(self.vgg_mdoel, True)
            self.vgg_mdoel.eval()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), 
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), 
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A  = input['A'].to(self.device)
        self.cond_B = input['cond_B'].to(self.device).float().cuda()
        #if self.isTrain:
        self.real_B  = input['B'].to(self.device)
        self.cond_A = input['cond_A'].to(self.device).float().cuda()
        #else:
        self.image_paths = input['A_paths']

    def _set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def _channel_loss(self, x, y):
        return self.criterionRec1(x[:,0,:,:], y[:,0,:,:]) + self.criterionRec2(x[:,0,:,:], y[:,0,:,:]) + \
               self.criterionRec1(x[:,1,:,:], y[:,1,:,:]) + self.criterionRec2(x[:,1,:,:], y[:,1,:,:]) + \
               self.criterionRec1(x[:,2,:,:], y[:,1,:,:]) + self.criterionRec2(x[:,2,:,:], y[:,2,:,:])

    def forward(self):
        # forward reconstruction: G(A, S_B) -> B
        self.fake_B = self.netG(torch.cat([self.real_A, self.cond_B],dim=1))
        if self.isTrain:
            # G(B, S_A) -> A
            self.fake_A = self.netG(torch.cat([self.real_B, self.cond_A],dim=1))
            
            # cycle reconstruction: G(G(B, S_A), S_B) -> B and G(G(A, S_B), S_A) -> A
            self.cycle_B = self.netG(torch.cat([self.fake_A, self.cond_B],dim=1))
            self.cycle_A = self.netG(torch.cat([self.fake_B, self.cond_A],dim=1))
            
            # identity: G(A, S_A) -> A and G(B, S_B) -> B
            #self.idt_A = self.netG(torch.cat([self.real_A, self.cond_A],dim=1))
            #self.idt_B = self.netG(torch.cat([self.real_B, self.cond_B],dim=1))

            self.feature_real_B = self.vgg_mdoel(F.interpolate(self.real_B, size=(224, 224), mode='bilinear', align_corners=False))
            self.feature_fake_B = self.vgg_mdoel(F.interpolate(self.fake_B, size=(224, 224), mode='bilinear', align_corners=False))

            self.feature_real_A = self.vgg_mdoel(F.interpolate(self.real_A, size=(224, 224), mode='bilinear', align_corners=False))
            self.feature_fake_A = self.vgg_mdoel(F.interpolate(self.fake_A, size=(224, 224), mode='bilinear', align_corners=False))

    def backward_D(self):
        """Calculate the loss for Discriminator D"""
        # Real B
        loss_D_real_B = self.criterionGAN(self.netD(self.real_B), True)
        # Fake B
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_fake_B = self.criterionGAN(self.netD(self.fake_B.detach()), False)
        # Real A
        loss_D_real_A = self.criterionGAN(self.netD(self.real_A), True)
        # Fake A
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_fake_A = self.criterionGAN(self.netD(self.fake_A.detach()), False)
        
        self.loss_D = (loss_D_real_A + loss_D_fake_A + loss_D_real_B + loss_D_fake_B)
        self.loss_D.backward()

    def backward_G(self):
        """Calculate the loss for generators G"""
        # GAN loss D(G(*))
        loss_G_B = self.criterionGAN(self.netD(self.fake_B), True)
        loss_G_A = self.criterionGAN(self.netD(self.fake_A), True)
        self.loss_GAB = (loss_G_B + loss_G_A) * self.opt.lambda_adv

        # forward reconstruction loss
        loss_rec_B = self.criterionRec1(self.real_B, self.fake_B) + self.criterionRec2(self.real_B, self.fake_B)
        loss_rec_A = self.criterionRec1(self.real_A, self.fake_A) + self.criterionRec2(self.real_A, self.fake_A)
        self.loss_rec = (loss_rec_B + loss_rec_A) * self.opt.lambda_rec

        # cycle reconstruction loss
        loss_cycle_A = self.criterionRec1(self.real_A, self.cycle_A)
        loss_cycle_B = self.criterionRec1(self.real_B, self.cycle_B)
        self.loss_cycle = (loss_cycle_A + loss_cycle_B) * self.opt.lambda_cycle

        # self-reconstruction (identity) loss
        loss_idt_B = self.criterionRec1(self.feature_real_B, self.feature_fake_B)
        loss_idt_A = self.criterionRec1(self.feature_real_A, self.feature_fake_A)
        #loss_idt_A = self.criterionRec1(self.real_A, self.idt_A)
        self.loss_idt = (loss_idt_B + loss_idt_A) * self.opt.lambda_idt

        # color loss
        loss_color_B = self._channel_loss(self.real_B, self.fake_B)
        loss_color_A = self._channel_loss(self.real_A, self.fake_A)
        self.loss_color = (loss_color_B + loss_color_A)* self.opt.lambda_color

        self.loss_G = self.loss_GAB + self.loss_rec + self.loss_cycle + self.loss_idt + self.loss_color
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()                # compute fake images and reconstruction images.
        self.set_requires_grad([self.netD], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()             # calculate gradients for G
        self.optimizer_G.step()       # update G's weights
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()             # calculate gradients for D
        self.optimizer_D.step()       # update D's weights
