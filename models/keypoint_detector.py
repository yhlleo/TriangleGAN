# Author: Yahui Liu <yahui.liu@unitn.it>

from torch import nn
import torch
import numpy as np
from skimage.draw import circle
import matplotlib.pyplot as plt

'''
Visualize heatmaps of encoded features based on the project: https://github.com/AliaksandrSiarohin/monkey-net
'''

def matrix_inverse(batch_of_matrix, eps=0):
    if eps != 0:
        init_shape = batch_of_matrix.shape
        a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
        b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
        c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
        d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

        det = a * d - b * c
        out = torch.cat([d, -b, -c, a], dim=-1)
        eps = torch.tensor(eps).type(out.type())
        out /= det.max(eps)

        return out.view(init_shape)
    else:
        b_mat = batch_of_matrix
        eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
        b_inv, _ = torch.gesv(eye, b_mat)
        return b_inv

def smallest_singular(batch_of_matrix):
    a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
    b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
    c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
    d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

    s1 = a ** 2 + b ** 2 + c ** 2 + d ** 2
    s2 = (a ** 2 + b ** 2 - c ** 2 - d ** 2) ** 2
    s2 = torch.sqrt(s2 + 4 * (a * c + b * d) ** 2)

    norm = torch.sqrt((s1 - s2) / 2)
    return norm

def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
    meshed = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2)
    return meshed

def gaussion2kp(heatmap, kp_variance='matrix', clip_variance=None):
    """
    Extract the mean and the variance from a heatmap
    """
    shape = heatmap.shape # [N, C, H, W]
    #adding small eps to avoid 'nan' in variance
    heatmap = heatmap.unsqueeze(-1) + 1e-7 # [1, C, H, W, 1]
    grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze(0).unsqueeze(0) # [1, 1, H, W, 2]
    mean = (heatmap * grid).sum(dim=(2, 3)) # [1, C, 2]

    #kp = {'mean': mean.permute(0, 2, 1, 3)}
    kp = {'mean': mean}

    if kp_variance == 'matrix':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2) # [1, 1, H, W, 2] - [1, C, 1, 1, 2]
        var = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2)) # [1, C, H, W, 2, 2]
        var = var * heatmap.unsqueeze(-1) # [1, C, H, W, 1, 1]
        var = var.sum(dim=(2, 3)) # [1, C, 2, 2]
        #var = var.permute(0, 2, 1, 3, 4)
        if clip_variance:
            min_norm = torch.tensor(clip_variance).type(var.type())
            sg = smallest_singular(var).unsqueeze(-1)
            var = torch.max(min_norm, sg) * var / sg
        kp['var'] = var
    elif kp_variance == 'single':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        var = mean_sub ** 2
        var = var * heatmap
        var = var.sum(dim=(2, 2))
        var = var.mean(dim=-1, keepdim=True)
        var = var.unsqueeze(-1)
        #var = var.permute(0, 2, 1, 3, 4)
        kp['var'] = var
    return kp

def kp2gaussion(kp, spatial_size, kp_variance='matrix'):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['mean'] # [1, C, 2]
    coordinate_grid = make_coordinate_grid(spatial_size, mean.type()) # [H, W, 2]
    number_of_leading_dimensions = len(mean.shape) - 1 # 2
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape # [1, 1, H, W, 2]

    coordinate_grid = coordinate_grid.view(*shape) # [1, 1, H, W, 2]
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1) # [1, C, 1, 1, 1]
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2) # [1, C, 1, 1, 2]
    mean = mean.view(*shape) 

    mean_sub = (coordinate_grid - mean) # [1, C, H, W, 2]
    if kp_variance == 'matrix':
        var = kp['var'] # [1, C, 2, 2]
        inv_var = matrix_inverse(var) # [1, C, 2, 2]
        shape = inv_var.shape[:number_of_leading_dimensions] + (1, 1, 2, 2) # [1, C, 1, 1, 2, 2]
        inv_var = inv_var.view(*shape)
        under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), inv_var), mean_sub.unsqueeze(-1)) # [1, C, H, W, 1, 1]
        under_exp = under_exp.squeeze(-1).squeeze(-1) # [1, C, H, W]
        out = torch.exp(-0.5 * under_exp)
    elif kp_variance == 'single':
        out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp['var'])
    else:
        out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

def normalize_heatmap(heatmap, norm_const='sum'):
    if norm_const == "sum":
        heatmap_shape = heatmap.shape
        heatmap = heatmap.view(heatmap_shape[0], heatmap_shape[1], -1)
        heatmap = heatmap / heatmap.sum(dim=2, keepdim=True)
        return heatmap.view(*heatmap_shape)
    else:
        return heatmap / norm_const

def keypoint_detector(heatmap, kp_variance=0.01, clip_variance=0.001):
    shape = heatmap.shape
    kp = gaussion2kp(heatmap, kp_variance, clip_variance)
    kp_heatmap = kp2gaussion(kp, shape[2:], kp_variance)
    return kp, normalize_heatmap(kp_heatmap)

def visualizer(src_img, kp_array, kp_size=2, colormap='gist_rainbow'):
    # torch tensor to numpy
    kp_array = kp_array['mean'].data.cpu().numpy()
    src_img = src_img.data.cpu().numpy()
    # draw key point into image
    src_img = np.transpose(src_img, [0, 2, 3, 1])
    cmap = plt.get_cmap(colormap)
    spatial_size = np.array(src_img.shape[1:3])[np.newaxis, np.newaxis]
    #print(kp_array)
    kp_array = spatial_size * (kp_array + 1) / 2
    num_kp = kp_array.shape[1]
    for kp_ind, kp in enumerate(kp_array[0]):
        rr, cc = circle(kp[1], kp[0], kp_size, shape=src_img.shape[1:3])
        src_img[0][rr, cc] = np.array(cmap(float(kp_ind) / num_kp))[:3] - 0.5
    # numpy to tensor
    return torch.from_numpy(np.transpose(src_img, [0,3,1,2]))
