import timeit
import torch

from hooks import *

import warnings
warnings.filterwarnings("ignore")




def recon_meta_grad_linear(input_train, output_grad_train, test_grad, device="cuda"):
    """
    :param input_train:
    :param input_test:
    :param output_grad_train:
    :param output_grad_test:
    :return:
    """
    if device == "cuda":
        input_train = input_train.cuda()
        output_grad_train = output_grad_train.cuda()
        test_grad = test_grad.cuda()
    elif device == "cpu":
        input_train = input_train.cpu()
        output_grad_train = output_grad_train.cpu()
        test_grad = test_grad.cpu()
    else:
        raise NotImplementedError("Unkown device type {}".format(device))


    if len(input_train.shape) == 2:
        test_grad = test_grad.squeeze()
        recon_train_grad = torch.einsum('bp,bq->bpq', input_train, output_grad_train)
        meta_grad = torch.mul(recon_train_grad, test_grad.T).sum((-1, -2))
    else:
        recon_train_grad = torch.einsum('blp,blq->bpq', input_train, output_grad_train)
        meta_grad = torch.mul(recon_train_grad, test_grad.T).sum((-1, -2))

    return meta_grad


def _recon_meta_grad_conv2d(train_feats_dict, train_outputs_grad_dict, test_grad_dict):
    meta_grad = 0

    for key in train_feats_dict.keys():
        _input_train_slice, _output_grad_train_slice = train_feats_dict[key], train_outputs_grad_dict[key]
        _test_grad_slice = test_grad_dict[key]

        _recon_train_grad = torch.einsum("bxmn,bymn->byxmn", _input_train_slice, _output_grad_train_slice).sum((-1,-2))

        _meta_grad = torch.multiply(_recon_train_grad, _test_grad_slice).sum((1,2))

        meta_grad += _meta_grad

    return meta_grad


def recon_meta_grad_conv2d(input_train, output_grad_train, test_grad, device, **kwargs):
    """
    :param input_train:
    :param input_test:
    :param output_grad_train:
    :param output_grad_test:
    :return:
    """

    if device == "cuda":
        input_train = input_train.cuda()
        output_grad_train = output_grad_train.cuda()
        test_grad = test_grad.cuda()

    def pad_input(x):
        padding_dim = (x.shape[0], x.shape[1], x.shape[-2] + padding * 2, x.shape[-1] + padding * 2)
        if device == "cuda":
            pad_input = torch.zeros(padding_dim).cuda()
        elif device == "cpu":
            pad_input = torch.zeros(padding_dim)
        else:
            raise NotImplementedError("Unkown device type {}".format(device))
        pad_input[:, :, padding: -padding, padding: -padding] = x

        return pad_input

    if "padding" in kwargs:
        padding = kwargs["padding"]
        input_train = pad_input(input_train)
    kernel_size = kwargs["kernel_size"]

    if len(test_grad.shape) == 5:
        test_grad = test_grad.squeeze(0)

    train_feats_dict = {}
    train_outputs_grad_dict = {}
    test_grad_dict = {}
    for m in range(kernel_size):
        for n in range(kernel_size):
            t1, t2 = output_grad_train.shape[-2], output_grad_train.shape[-1]
            _input_train_slice = input_train[:, :, m:m + t1, n:n + t2]

            if device == "cuda":
                _output_grad_train_slice = torch.tensor(output_grad_train).cuda()
            elif device == "cpu":
                _output_grad_train_slice = torch.tensor(output_grad_train).cpu()
            else:
                raise NotImplementedError("Unkown device type {}".format(device))

            key = (m, n)
            train_feats_dict[key] = _input_train_slice
            train_outputs_grad_dict[key] = _output_grad_train_slice
            test_grad_dict[key] = test_grad[:, :, m, n]

    meta_grad = _recon_meta_grad_conv2d(train_feats_dict, train_outputs_grad_dict, test_grad_dict)

    return meta_grad


if __name__ == "__main__":

    m1 = torch.reshape(torch.arange(5), (1, 1, 5))
    m2 = torch.reshape(torch.arange(10), (2, 1, 5))

    print (m1)
    print (m2)
    print (torch.mul(m1, m2).sum((-1,-2)))