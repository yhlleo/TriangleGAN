import torch
import itertools
import torch.nn.functional as F
import numpy as np
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import keypoint_detector as kpd
from . import trianglegan_networks as trianglegan

class TriangleGANModel(BaseModel):
    """
    TriangleGAN
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
            parser.add_argument('--lambda_rec', type=float, default=100.0, help='weight for reconstruction loss (A -> ((y,r),z) -> B)')
            parser.add_argument('--lambda_cycle', type=float, default=10.0, help='weight for cycle reconstruction')
            parser.add_argument('--lambda_adv', type=float, default=2.0, help='weight for adversarial loss')
            parser.add_argument('--lambda_prob', type=float, default=1.0, help='weight for discriminal loss')
            parser.add_argument('--lambda_cls', type=float, default=1.0, help='weight for classification loss')
            parser.add_argument('--lambda_gp', type=float, default=0.0, help='weight for gradient penalty loss')
            parser.add_argument('--lambda_idt', type=float, default=10.0, help='use identity mapping')
            parser.add_argument('--lambda_tv', type=float, default=1e-5, help='weight for tv loss')
        return parser

    def __init__(self, opt):
        """Initialize the TriangleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D', 'cls', 'gp',
                           'GAB1', 'gcls1', 'rec1', 'cycle1', 'idt1', 'tv1', 
                           'GAB2', 'gcls2', 'rec2', 'cycle2', 'idt2', 'tv2']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>

        self.visual_names = ['real_A', 'real_B', 'cond_B', 'fake_B1_masked', 'fake_B2_mask', 'fake_B2_masked']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G1', 'G2', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G1', 'G2']

        # define networks (both Generators and discriminators)
        # encode image to features
        self.netG1 = trianglegan.define_GTriangleGAN(opt.input_nc,
                                                     opt.output_nc,
                                                     opt.ngf,
                                                     opt.vdim,
                                                     opt.cond_dim,
                                                     opt.num_kp,
                                                     opt.use_kpd,
                                                     False,
                                                     opt.norm,
                                                     not opt.no_dropout,
                                                     opt.init_type,
                                                     opt.init_gain,
                                                     self.gpu_ids)

        self.netG2 = trianglegan.define_GTriangleGAN(opt.input_nc,
                                                     opt.output_nc,
                                                     opt.ngf,
                                                     opt.vdim,
                                                     opt.cond_dim,
                                                     opt.num_kp,
                                                     opt.use_kpd,
                                                     True,
                                                     opt.norm,
                                                     not opt.no_dropout,
                                                     opt.init_type,
                                                     opt.init_gain,
                                                     self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD = trianglegan.define_DStar(opt.output_nc+1, 
                                                 opt.ndf,
                                                 opt.n_layers_D, 
                                                 opt.vdim,
                                                 opt.load_size,
                                                 opt.norm, 
                                                 opt.init_type, 
                                                 opt.init_gain, 
                                                 self.gpu_ids)

        if self.isTrain:
            # create image buffer to store previously generated images
            self.fake_B1_masked_pool = ImagePool(opt.pool_size) 
            self.fake_B2_masked_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionRec1 = torch.nn.L1Loss()
            self.criterionRec2 = torch.nn.MSELoss()
            self.criterionSim = torch.nn.MSELoss()
            self.criterionCls = F.cross_entropy
            self.criterionSmooth = networks.compute_loss_smooth
            self.criterionNorm = torch.mean

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG1.parameters(),
                                                                self.netG2.parameters()), 
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
        self.RB = input['R_B'].to(self.device).float().cuda()
        self.cond_B = input['cond_B'].to(self.device).float().cuda()
        #if self.isTrain:
        self.real_B  = input['B'].to(self.device)
        self.RA = input['R_A'].to(self.device).float().cuda()
        self.cond_A = input['cond_A'].to(self.device).float().cuda()
        if self.isTrain:
            if self.opt.lambda_cls > 0:
                self.real_RA = torch.argmax(input['R_A'].to(self.device), dim=1).long().cuda()
                self.real_RB = torch.argmax(input['R_B'].to(self.device), dim=1).long().cuda()
        #else:

    def _inference(self, net, real_A, cond_B, RB, rec_A=None):
        kp, fake_B, fake_B_mask = net(real_A, cond_B, RB, rec_A)
        fake_B_masked = fake_B_mask * real_A + (1 - fake_B_mask) * fake_B
        return kp, fake_B, fake_B_mask, fake_B_masked

    def _forward_step1(self):
        # forward reconstruction: A + cond_B -> B
        self.kp1, self.fake_B1, self.fake_B1_mask, self.fake_B1_masked = self._inference(self.netG1, 
                                                                                         self.real_A, 
                                                                                         self.cond_B, 
                                                                                         self.RB)
        if self.isTrain:
            # cycle reconstruction: G1(A) + cond_A -> A
            _, self.rec_real_A1, _, self.rec_fake_A1_masked = self._inference(self.netG1, 
                                                                              self.fake_B1_masked,
                                                                              self.cond_A,
                                                                              self.RA)

            # self reconstruction (identity): A + cond_A -> A
            _, self.fake_A1_copy, _, self.fake_A1_copy_masked = self._inference(self.netG1, 
                                                                                self.real_A,
                                                                                self.cond_A,
                                                                                self.RA)
            # gradient penalty
            alpha = torch.rand(self.real_A.size(0), 1, 1, 1).to(self.device)
            self.B1_hat = (alpha * self.real_B.data + (1 - alpha) * self.fake_B1_masked.data).requires_grad_(True)
    
    def _forward_step2(self):
        self.kp2, self.fake_B2, self.fake_B2_mask, self.fake_B2_masked = self._inference(self.netG2,
                                                                                         self.real_A,
                                                                                         self.cond_B,
                                                                                         self.RB,
                                                                                         self.fake_B1_masked.detach())
        if self.isTrain:
            # cycle reconstruction: G1(A) + cond_A -> A
            _, self.rec_real_A2, _, self.rec_fake_A2_masked = self._inference(self.netG2, 
                                                                              self.fake_B2_masked,
                                                                              self.cond_A,
                                                                              self.RA,
                                                                              self.rec_fake_A1_masked.detach())
            # self reconstruction (identity): A + cond_A -> A
            _, self.fake_A2_copy, _, self.fake_A2_copy_masked = self._inference(self.netG2,
                                                                                self.real_A,
                                                                                self.cond_A,
                                                                                self.RA,
                                                                                self.fake_A1_copy_masked.detach())
            # gradient penalty
            alpha = torch.rand(self.real_A.size(0), 1, 1, 1).to(self.device)
            self.B2_hat = (alpha * self.real_B.data + (1 - alpha) * self.fake_B2_masked.data).requires_grad_(True)
        else:
            if self.opt.roll_num > 1:
                for i in range(1, self.opt.roll_num):
                    self.kp2, self.fake_B2, self.fake_B2_mask, self.fake_B2_masked = self._inference(self.netG2, 
                                                                                                     self.real_A,
                                                                                                     self.cond_B,
                                                                                                     self.RB,
                                                                                                     self.fake_B2_masked.detach())
        self.fake_B2_mask = (self.fake_B2_mask-0.5)/0.5 # for visualization

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self._forward_step1()
        self._forward_step2()
        if not self.isTrain and self.opt.draw_kp:
            self.fake_B1_masked = kpd.visualizer(self.fake_B1_masked, self.kp1)
            self.fake_B2_masked = kpd.visualizer(self.fake_B2_masked, self.kp2)

    def backward_D(self):
        # Discriminator loss
        # Real B
        out_src_real_B, out_cls_real_B =  self.netD(torch.cat([self.real_B, self.cond_B], dim=1))
        loss_D_real_B = self.criterionGAN(out_src_real_B, True)
        # Fake B1
        fake_B1_masked = self.fake_B1_masked_pool.query(self.fake_B1_masked)
        out_src_fake_B1, _ = self.netD(torch.cat([fake_B1_masked, self.cond_B], dim=1).detach())
        loss_D_fake_B1 = self.criterionGAN(out_src_fake_B1, False)
        # Fake B2
        fake_B2_masked = self.fake_B2_masked_pool.query(self.fake_B2_masked)
        out_src_fake_B2, _ = self.netD(torch.cat([fake_B2_masked, self.cond_B], dim=1).detach())
        loss_D_fake_B2 = self.criterionGAN(out_src_fake_B2, False)
        self.loss_D = (loss_D_real_B*2 + loss_D_fake_B1 + loss_D_fake_B2) * 0.5 * self.opt.lambda_prob
        
        # gradient penalty loss
        if self.opt.lambda_gp > 0.0:
            out_src_hat_B1, _ = self.netD(torch.cat([self.B1_hat, self.cond_B], dim=1))
            loss_gp_B1 = networks.gradient_penalty(out_src_hat_B1, self.B1_hat, self.device)
            out_src_hat_B2, _ = self.netD(torch.cat([self.B2_hat, self.cond_B], dim=1))
            loss_gp_B2 = networks.gradient_penalty(out_src_hat_B2, self.B2_hat, self.device)
            self.loss_gp = (loss_gp_B1 + loss_gp_B2) * 0.5 * self.opt.lambda_gp
        else:
            self.loss_gp = 0.0

        # classification loss
        if self.opt.lambda_cls> 0:
            self.loss_cls = self.criterionCls(out_cls_real_B, self.real_RB) * self.opt.lambda_cls
        else:
            self.loss_cls = 0.0

        self.loss_D_all = self.loss_D + self.loss_gp + self.loss_cls
        self.loss_D_all.backward()

    def backward_G1(self):
        """Calculate the loss for generators G"""
        # GAN loss
        out_src_fake_B, out_cls_fake_B = self.netD(torch.cat([self.fake_B1_masked, self.cond_B], dim=1))
        self.loss_GAB1 = self.criterionGAN(out_src_fake_B, True) * self.opt.lambda_adv
        self.loss_gcls1 = self.criterionCls(out_cls_fake_B, self.real_RB) * self.opt.lambda_cls

        # Forward reconstruction loss:
        self.loss_rec1 = self.criterionRec1(self.fake_B1_masked, self.real_B) * self.opt.lambda_rec

        # Forward identity loss
        self.loss_idt1 = self.criterionRec1(self.fake_A1_copy_masked, self.real_A) * self.opt.lambda_idt

        # cycle reconstruction loss
        self.loss_cycle1 = self.criterionRec1(self.rec_fake_A1_masked, self.real_A) * self.opt.lambda_cycle

        # smooth loss for attention mask
        loss_tv1 = self.criterionSmooth(self.fake_B1) + self.criterionSmooth(self.fake_A1_copy)
        self.loss_tv1 = loss_tv1 * 0.5 * self.opt.lambda_tv 

        # combined loss and calculate gradients
        self.loss_G1 = self.loss_GAB1 + self.loss_gcls1 + self.loss_rec1 + self.loss_cycle1 + self.loss_idt1 * self.loss_tv1
        self.loss_G1.backward()

    def backward_G2(self):
        """Calculate the loss for generators G"""
        # GAN loss
        out_src_fake_B, out_cls_fake_B = self.netD(torch.cat([self.fake_B2_masked, self.cond_B], dim=1))
        self.loss_GAB2 = self.criterionGAN(out_src_fake_B, True) * self.opt.lambda_adv
        self.loss_gcls2 = self.criterionCls(out_cls_fake_B, self.real_RB) * self.opt.lambda_cls

        # Forward reconstruction loss:
        self.loss_rec2 = self.criterionRec1(self.fake_B2_masked, self.real_B) * self.opt.lambda_rec

        # Forward identity loss
        self.loss_idt2 = self.criterionRec1(self.fake_A2_copy_masked, self.real_A) * self.opt.lambda_idt

        # cycle reconstruction loss
        self.loss_cycle2 = self.criterionRec1(self.rec_fake_A2_masked, self.real_A) * self.opt.lambda_cycle

        # smooth loss for attention mask
        loss_tv2 = self.criterionSmooth(self.fake_B2) + self.criterionSmooth(self.fake_A2_copy)
        self.loss_tv2 = loss_tv2 * 0.5 * self.opt.lambda_tv 

        # combined loss and calculate gradients
        self.loss_G2 = self.loss_GAB2 + self.loss_gcls2 + self.loss_rec2 + self.loss_cycle2 + self.loss_idt2 * self.loss_tv2
        self.loss_G2.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()                # compute fake images and reconstruction images.
        self.set_requires_grad([self.netD], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G1()            # calculate gradients for G1
        self.backward_G2()            # calculate gradients for G2
        self.optimizer_G.step()       # update G's weights
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()             # calculate gradients for D
        self.optimizer_D.step()       # update D's weights
