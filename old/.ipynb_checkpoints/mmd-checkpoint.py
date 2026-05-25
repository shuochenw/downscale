import torch
import numpy as np
from torch.autograd import Variable

# --- MMD Loss ---
def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(x.size(0)) + int(y.size(0))
    total = torch.cat([x, y], dim=0)
    L2_distance = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    return sum(kernel_val)

def compute_mmd(x, y):
    batch_size = x.size(0)
    kernels = gaussian_kernel(x, y)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    return torch.mean(XX + YY - XY - YX)