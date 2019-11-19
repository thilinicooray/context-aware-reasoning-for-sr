import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from lib.fc import FCNet


class Attention(nn.Module):
    '''
    Calculates attention for each region of the image based on the query
    Reused code from original source https://github.com/hengyuan-hu/bottom-up-attention-vqa
    '''
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)

        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits
