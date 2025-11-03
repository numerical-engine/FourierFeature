import torch
import torch.nn as nn
import math

class Gaussian(nn.Module):
    def __init__(self, in_features:int, num_mode:int, scale:float)->None:
        super().__init__()
        self.B = nn.Parameter(2.*math.pi*torch.randn(in_features, num_mode)*scale, requires_grad = False)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return torch.cat((torch.cos(x@self.B), torch.sin(x@self.B)), dim = 1)