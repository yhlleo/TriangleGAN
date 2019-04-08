# Fréchet Resnet Distance (FRD) 
# Based on: https://ww2.mathworks.cn/matlabcentral/fileexchange/31922-discrete-frechet-distance
import torch
from torchvision.models import resnet50
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np

#// TODO: RecursionError: maximum recursion depth exceeded in comparison
# Runing the matlab code to test FRD scores:
# See: calculate_FRD.m
def dist(p, q, i, j, CA, dfcn):
    if CA[i,j] > -1.0:
        pass
    elif i==0 and j==0:
        CA[i, j] = dfcn(p[0], q[0])
    elif i>0 and j==0:
        CA[i,j] = max(dist(p,q,i-1,j,CA,dfcn), dfcn(p[i], q[0]))
    elif i==0 and j>0:
        CA[i,j] = max(dist(p,q,i,j-1,CA,dfcn), dfcn(p[0], q[j]))
    elif i>0 and j>0:
        CA[i,j] = max(min([dist(p,q,i-1,j,CA,dfcn), 
                           dist(p,q,i-1,j-1,CA,dfcn), 
                           dist(p,q,i,j-1,CA,dfcn)]), dfcn(p[i], q[j]))
    return CA[i,j]

def discrete_frechet_dist(p, q, dfcn=None):
    dfcn = dfcn if dfcn is not None else lambda a, b: np.sqrt(np.sum(np.square(a-b)))
    assert p.shape == q.shape, 'shape {} is not equal to {}'.format(p.shape, q.shape)
    i,j = p.shape[0], q.shape[0]
    CA = np.ones((i,j)) * -1.0
    return dist(p,q,i-1,j-1,CA, dfcn)

def model_forward(model, batch, dims):
    pred = model(batch)
    return pred.view(-1, dims).cpu().data.numpy()

def cal_frechet_resnet_distance(data1, data2, dims=1000, use_cuda=True):
    model = resnet50(pretrained=True)
    if use_cuda:
        model.cuda()
    model.eval()

    scores1, scores2 = [], []
    for batch1, batch2 in zip(data1, data2):
        pred1 = model_forward(model, batch1, dims)
        scores1.append(pred1)

        pred2 = model_forward(model, batch2, dims)
        scores2.append(pred2)
     
    acts1 = np.concatenate(scores1, axis=0)
    acts2 = np.concatenate(scores2, axis=0)

    assert acts1.shape == acts2.shape, 'shape {} is not equal to {}.'.format(acts1.shape, acts2.shape)
    frd_scores = []
    for i in range(acts1.shape[0]):
        frd_scores.append(discrete_frechet_dist(acts1[i], acts2[i]))
    return np.mean(frd_scores)