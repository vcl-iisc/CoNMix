import numpy as np
import sys
from collections import Counter
import torch
import torch.nn.functional as F

from sklearn.utils import axis0_safe_slice
np.set_printoptions(threshold=sys.maxsize)

def plr(prev_pseudo_labels, pseudo_labels, soft_output, class_num,  alpha=0.9):

    consensus=torch.zeros((class_num, class_num))
    for i in range(class_num):
        index_i = np.where(prev_pseudo_labels == i)[0]
        for j in range(class_num):
            index_j = np.where(pseudo_labels == j)[0]
            # print(index_i.shape, index_j.shape)
            intersect = np.intersect1d(index_i, index_j)
            union = np.union1d(index_i, index_j)
            consensus[i][j] = len(intersect)/(len(union)+1e-8)
    
    consensus = F.softmax(consensus, dim=1)
    # print('consensus: ', consensus.shape)
    # print(consensus)
    prev_pseudo_labels = torch.unsqueeze(torch.from_numpy(prev_pseudo_labels), dim=1)
    pseudo_labels = torch.unsqueeze(torch.from_numpy(pseudo_labels), dim=1)

    prop_prev_pl = torch.matmul(torch.from_numpy(soft_output), consensus)
    # print(prop_prev_pl.shape)

    refined = torch.add(alpha*pseudo_labels, (1-alpha)*prop_prev_pl)
    refined = F.softmax(refined, dim=1)

    return refined.numpy()