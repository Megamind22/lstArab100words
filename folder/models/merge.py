import torch.nn as nn
import torch.nn.functional as F
import torch


class mergeModels(nn.Module):
    def __init__(self, modelA, modelB):
        super(mergeModels, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        x1 = self.modelB(x)
        x2 = self.modelA(x1)
        #x = torch.cat(x2), dim=1) 
        return x2
