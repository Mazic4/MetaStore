import numpy as np

import torch

from torch.autograd.functional import jacobian

import warnings
warnings.filterwarnings("ignore")


def save_backward_hooks(self, grad_input, grad_output):
    torch.save(grad_output[0], "./artifacts/_cache/_{}_output_grad.pt".format(self.name))


def save_forward_hooks(self, input, output):
    torch.save(input[0], "./artifacts/_cache/_{}_input.pt".format(self.name))


def save_param_grad_hook(self, grad):

    if len(grad.shape) != 2:
        return

    _input = np.load("_{}_input.npy".format(self.name))
    print("The shape of _input is: ", _input.shape)

    _output_grad = np.load("_{}_output_grad.npy".format(self.name))
    print("The shape of _output_grad is: ", _output_grad.shape)

    retrieve_grad = np.dot(_output_grad.T, _input)

    print(np.allclose(grad, retrieve_grad))
