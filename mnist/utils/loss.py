import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model import NumClass


class EBEMLoss(nn.Module):
    def __init__(self):
        super(EBEMLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, P_matrix): # P_matrix: size = batch * K
        loss, N_sample = 0.0, x.size(0)
        for k in range(NumClass):
            target = (k * torch.ones(N_sample).to(x.device)).long()
            loss += torch.sum(P_matrix[:, k] * self.criterion(x, target) ) / N_sample
        return loss
