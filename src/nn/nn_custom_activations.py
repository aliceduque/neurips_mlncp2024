import torch
import torch.nn as nn


class erf(nn.Module):
    def forward(self, x):
        return torch.erf(x)

class ReLU_bound(nn.Module):
    def forward(self, x):
        y = torch.zeros_like(x)
        a = 8
        y[(x <= 0)] = 0
        y[(x > 0) & (x < a)] = x[(x > 0) & (x < a)]/a
        y[(x >= a)] = 1
        return y

class ReLU_bound_symm(nn.Module):
    def forward(self, x):
        y = torch.zeros_like(x)
        a = 8
        y[(x <= -(a/2))] = 0
        y[(x > -(a/2)) & (x < (a/2))] = x[(x > (-a/2)) & (x < (a/2))]/a + 0.5
        y[(x >= (a/2))] = 1
        return y
    
class Sigm_shift(nn.Module):
    def forward(self, x):
        y = torch.sigmoid(x-4)
        return y