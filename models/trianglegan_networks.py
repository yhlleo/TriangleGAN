import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from .networks import get_norm_layer, init_net, ResnetBlock
from . import keypoint_detector as kpd

# Our GAN
#-------------------------------------------------------------------------------------
class TriangleGANGenerator(nn.Module):
    def __init__(self, in_nc, out_nc, ngf, vdim=11, cond_dim=1, num_kp=64, 
                 norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, 
                 padding_type='reflect', use_kpd=0, temperature=0.1, kp_variance=0.01,
                 clip_variance=0.001, multisteps=False):
        super(TriangleGANGenerator, self).__init__()
        assert(n_blocks > 0)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.ngf = ngf
        self.norm_layer = norm_layer
        self.use_bias = use_bias
        self.use_dropout = use_dropout
        self.n_blocks = n_blocks
        self.padding_type = padding_type
        self.max_nc = 512
        self.vdim = vdim
        self.cond_dim = cond_dim
        self.use_kpd = use_kpd
        self.num_kp = num_kp if self.use_kpd else ngf
        self.rec_nc = 3 if multisteps else 0
        
        self.kp_detector = kpd.keypoint_detector
        self.temperature = temperature
        self.kp_variance = kp_variance
        self.clip_variance = clip_variance
        
        block_expansion = ngf*4
        
        self.image_content = nn.Sequential(*self._img_encoder(in_nc, up_nc=True))    # out_nc: ngf*4
        self.condition = nn.Sequential(*self._img_encoder(self.vdim+self.cond_dim+self.rec_nc, self.num_kp, up_nc=False))
        self.resblock = nn.Sequential(*self._resblock(block_expansion+self.num_kp, block_expansion))
        self.decode = nn.Sequential(*self._decoder(block_expansion+self.num_kp, block_expansion))

        # image reconstruction part 
        model = [nn.Conv2d(ngf, out_nc, kernel_size=7, stride=1, padding=3, bias=False),
                 nn.Tanh()]
        self.img_rec = nn.Sequential(*model)
        # attention mask part 
        model = [nn.Conv2d(ngf, 1, kernel_size=7, stride=1, padding=3, bias=False),
                 nn.Sigmoid()]
        self.attetion_reg = nn.Sequential(*model)

    def _img_encoder(self, in_nc, out_nc=None, up_nc=False):
        if out_nc is None:
            out_nc = self.ngf

        model = [nn.Conv2d(in_nc, out_nc, kernel_size=7, stride=1, padding=3, bias=self.use_bias),
                 self.norm_layer(out_nc),
                 nn.ReLU(True)]
        cur_in_nc, cur_out_nc = out_nc, out_nc
        for i in range(2):
            cur_out_nc = cur_out_nc*2 if up_nc and (cur_out_nc<self.max_nc) else cur_out_nc
            model += [nn.Conv2d(cur_in_nc, cur_out_nc, kernel_size=3, stride=2, padding=1, bias=self.use_bias),
                      self.norm_layer(cur_out_nc, affine=True),
                      nn.ReLU(True)]
            cur_in_nc = cur_out_nc
        return model

    def _resblock(self, in_nc, out_nc):
        model = [nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
                 self.norm_layer(out_nc, affine=True),
                 nn.ReLU(True)]
        # residual blocks
        for i in range(self.n_blocks):
            model += [ResnetBlock(out_nc, padding_type=self.padding_type, norm_layer=self.norm_layer, 
                                  use_dropout=self.use_dropout, use_bias=self.use_bias)]
        return model

    def _decoder(self, in_nc, out_nc):
        # combine the features and reconstruct the features
        model = [nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
                 self.norm_layer(out_nc, affine=True),
                 nn.ReLU(True)]
        # Up-Sampling
        for i in range(2):
            model += [nn.ConvTranspose2d(out_nc, out_nc//2, kernel_size=3, stride=2, 
                                         padding=1, output_padding=1, bias=False),
                      self.norm_layer(out_nc//2, affine=True),
                      nn.ReLU(True)]
            out_nc = out_nc // 2
        return model

    def forward(self, x, cond, r, rec_x=None):
        #idx = torch.argmax(r)
        in_r = r.unsqueeze(2).unsqueeze(3).expand(r.size(0), r.size(1), x.size(2), x.size(3))
        image_features = self.image_content(x)
        
        if rec_x is None:
            condition_features = self.condition(torch.cat([cond, in_r], dim=1))
        else:
            condition_features = self.condition(torch.cat([rec_x, cond, in_r], dim=1))

        final_shape = condition_features.shape
        heatmap = condition_features.view(final_shape[0], final_shape[1], -1).clone()
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)
        kp, heatmap = self.kp_detector(heatmap, self.kp_variance, self.clip_variance)
        if self.use_kpd:
            condition_features = heatmap

        resblock_out = self.resblock(torch.cat([image_features, condition_features], dim=1))
        decode_out = self.decode(torch.cat([resblock_out, condition_features], dim=1))
        return kp, self.img_rec(decode_out), self.attetion_reg(decode_out)

def define_GTriangleGAN(in_nc, 
                       out_nc,
                       ngf,
                       vdim,
                       cond_dim,
                       num_kp,
                       use_kpd=0,
                       multisteps=False,
                       norm='instance', 
                       use_dropout=False, 
                       init_type='normal',
                       init_gain=0.02, 
                       gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net = TriangleGANGenerator(in_nc, out_nc, ngf=ngf, vdim=vdim, cond_dim=cond_dim, num_kp=num_kp,
                              norm_layer=norm_layer, use_dropout=use_dropout, use_kpd=use_kpd,
                              multisteps=multisteps)
    return init_net(net, init_type, init_gain, gpu_ids)


# Discriminator of StarGAN and GANimation
#-------------------------------------------------------------------------------------
class StarGANDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=6, vdim=11, norm_layer=nn.InstanceNorm2d, img_size=256):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of hidden conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(StarGANDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.01)]
        cur_dim = ndf
        for n in range(1, n_layers):  # gradually increase the number of filters
            sequence += [
                nn.Conv2d(cur_dim, cur_dim*2, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                nn.LeakyReLU(0.01)]
            cur_dim = cur_dim * 2

        self.main = nn.Sequential(*sequence)
        self.output_src = nn.Conv2d(cur_dim, 1, kernel_size=kw, stride=1, 
                                    padding=padw, bias=False)         # output 1 channel prediction map

        self.output_cls = nn.Conv2d(cur_dim, vdim, kernel_size=img_size//(2**n_layers), 
                                    stride=1, padding=0, bias=False)  # output vdim channel predicted vector

    def forward(self, x):
        features = self.main(x)
        out_src = self.output_src(features)
        out_cls = self.output_cls(features)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

def define_DStar(input_nc, ndf, n_layers_D=3, vdim=11, img_size=256, norm='instance', 
                 init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net = StarGANDiscriminator(input_nc, ndf, n_layers_D, vdim, norm_layer, img_size)
    return init_net(net, init_type, init_gain, gpu_ids) 