import numpy as np

import torch

from .utils import svd

def grad_to_feats_linear(grad: np.array, n_pseudo_samples: int, random_state: int = 1) -> (torch.Tensor, torch.Tensor):

    U, Sigma, VT = svd(grad, num_components = n_pseudo_samples)

    feats = []
    outputs_grad = []

    for _idx_pseudo_sample in range(n_pseudo_samples):
        feats.append(VT[_idx_pseudo_sample].unsqueeze(0))
        outputs_grad.append(torch.multiply(U[:, _idx_pseudo_sample], Sigma[_idx_pseudo_sample]).unsqueeze(0))

    feats = torch.cat(feats, dim=0).unsqueeze(1)
    outputs_grad = torch.cat(outputs_grad, dim=0).unsqueeze(1)

    return feats, outputs_grad

def grad_to_feats_conv(grad: np.array, n_pseudo_samples: int, random_state: int = 1, **kwg) -> dict:

    kernel_size = kwg["kernel_size"]

    feats_dict = {}
    outputs_grad_dict = {}

    for m in range(kernel_size):
        for n in range(kernel_size):

            _grad_slices = grad[:, :, m, n]
            feat, output_grad = grad_to_feats_linear(_grad_slices, n_pseudo_samples, random_state=random_state)
            feats_dict[(m,n)] = feat.T.unsqueeze(0)
            outputs_grad_dict[(m,n)] = output_grad.T.unsqueeze(0)

    return feats_dict, outputs_grad_dict
