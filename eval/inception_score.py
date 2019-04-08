# Inception Score (IS)
# Based on: https://github.com/sbarratt/inception-score-pytorch

import math
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy.stats import entropy

def inception_score(data_generator, use_cuda=True, splits=10):
    model = inception_v3(pretrained=True)
    if use_cuda:
        model.cuda()
    model.eval()

    scores = []
    for batch in data_generator:
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        # Scale from range (0, 1) to range (-1, 1)
        s = model(batch)
        scores.append(F.softmax(s,dim=1).data.cpu().numpy())

    scores = np.concatenate(scores, 0)
    # Now compute the mean kl-div
    split_scores = []
    N = scores.shape[0]
    for k in range(splits):
        start = k * (N // splits)
        end = (k+1) * (N // splits) if (k+1) * (N // splits) < N else N
        part = scores[start:end, :]
        py = np.mean(part, axis=0)
        cur_scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            cur_scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(cur_scores)))

    return np.mean(split_scores)
