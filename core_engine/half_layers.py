import timeit

import torch
from hooks import *

import warnings
warnings.filterwarnings("ignore")

def half_bert_meta_grad_linear(input_train, output_grad_train, test_grad, device):
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
    elif device == "cpu":
        input_train = input_train.cpu()
        output_grad_train = output_grad_train.cpu()
    else:
        raise NotImplementedError("Unkown device type {}".format(device))

    if len(input_train.shape) == 2:
        test_grad = test_grad.squeeze()
        p1 = torch.tensordot(input_train, test_grad, dims=([1], [1]))
        p2 = torch.mul(p1, output_grad_train).sum(1)
    elif len(input_train.shape) == 3:
        p1 = torch.tensordot(input_train, test_grad, dims=([2], [1]))
        p2 = torch.mul(p1, output_grad_train).sum((1,2))
    else:
        raise ValueError("Num dim is larger than 3.")

    meta_grad = p2

    return meta_grad


def _half_meta_grad_conv2d(train_feats_dict, train_outputs_grad_dict, test_grad_dict):
    meta_grad = 0

    for key in train_feats_dict.keys():
        _input_train_slice, _output_grad_train_slice = train_feats_dict[key], train_outputs_grad_dict[key]
        _test_grad_slice = test_grad_dict[key]

        p1 = torch.tensordot(_input_train_slice, _test_grad_slice, dims=([1], [1]))
        p2 = torch.mul(p1.permute(0, 3, 1, 2), _output_grad_train_slice).sum((1,2,3))

        meta_grad += p2

    return meta_grad


def half_meta_grad_conv2d(input_train, output_grad_train, test_grad, device, **kwargs):
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

    _ = 0

    train_feats_dict = {}
    train_outputs_grad_dict = {}
    test_grad_dict = {}
    test_grad = test_grad.squeeze()
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

    meta_grad = _half_meta_grad_conv2d(train_feats_dict, train_outputs_grad_dict, test_grad_dict)

    return meta_grad


def half_meta_grad_linear(input_train, output_grad_train, test_grad, device):
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
    elif device == "cpu":
        input_train = input_train.cpu()
        output_grad_train = output_grad_train.cpu()
    else:
        raise NotImplementedError("Unkown device type {}".format(device))

    if len(input_train.shape) == 2:
        p1 = torch.mm(input_train, test_grad)
        p2 = torch.mul(p1, output_grad_train).sum(1)
    else:
        len_sentence = input_train.shape[1]
        input_train = input_train.reshape(-1, input_train[-1])
        output_grad_train = output_grad_train.reshape(-1, output_grad_train[-1])
        p1 = torch.mm(input_train, test_grad)
        p2 = torch.mul(p1, output_grad_train).reshape(-1, len_sentence).sum((1,2))

    meta_grad = p2

    return meta_grad


if __name__ == "__main__":
    p1 = torch.arange(10).reshape(2, 5)
    p2 = torch.arange(10).reshape(2, 5)

    print (p1, p2)
    print (torch.einsum('bm, bn->b', p1, p2))