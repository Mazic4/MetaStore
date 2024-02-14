from utils.hooks import *

import warnings
warnings.filterwarnings("ignore")



def meta_grad_linear(input_train, input_test, output_grad_train, output_grad_test, device):

    """
    :param input_train:
    :param input_test:
    :param output_grad_train:
    :param output_grad_test:
    :return:
    """

    if len(input_test.shape) == 2:
        input_test = input_test.unsqueeze(0)
    if len(output_grad_test.shape) == 2:
        output_grad_test = output_grad_test.unsqueeze(0)

    if len(input_train.shape) == 2:
        input_train = torch.unsqueeze(input_train, dim=1)
    if len(output_grad_train.shape) == 2:
        output_grad_train = torch.unsqueeze(output_grad_train, dim=1)

    if device == "cuda":
        input_train, input_test = input_train.cuda(), input_test.cuda()
        output_grad_train, output_grad_test = output_grad_train.cuda(), output_grad_test.cuda()
    elif device == "cpu":
        input_train, input_test = input_train.cpu(), input_test.cpu()
        output_grad_train, output_grad_test = output_grad_train.cpu(), output_grad_test.cpu()
    else:
        raise NotImplementedError("Unkown device type {}".format(device))

    reconstructed_meta_gradients_p1 = torch.tensordot(input_train, input_test, dims=([2], [2]))
    reconstructed_meta_gradients_p2 = torch.tensordot(output_grad_train, output_grad_test, dims=([2],[2]))

    meta_grad = torch.multiply(reconstructed_meta_gradients_p1, reconstructed_meta_gradients_p2).sum((1,2,3))

    return meta_grad

def meta_grad_conv2d(input_train, input_test, output_grad_train, output_grad_test, device, **kwargs):
    if isinstance(output_grad_test, dict):
        return batch_meta_grad_conv2d(input_train, input_test, output_grad_train, output_grad_test, device, **kwargs)
    else:
        return raw_meta_grad_conv2d(input_train, input_test, output_grad_train, output_grad_test, device, **kwargs)

def raw_meta_grad_conv2d(input_train, input_test, output_grad_train, output_grad_test, device, **kwargs):

    """
    :param input_train:
    :param input_test:
    :param output_grad_train:
    :param output_grad_test:
    :return:
    """

    try:
        assert input_test.shape[0] == 1
    except:
        raise ValueError("The meta_grad_linear function only allows meta-gradient of 1 testing example")

    if len(input_test.shape) == 2:
        input_test = input_test.unsqueeze(1)
    if len(output_grad_test.shape) == 2:
        output_grad_test = output_grad_test.unsqueeze(1)

    if device == "cuda":
        input_train, input_test = input_train.cuda(), input_test.cuda()
        output_grad_train, output_grad_test = output_grad_train.cuda(), output_grad_test.cuda()
    elif device == "cpu":
        input_train, input_test = input_train.cpu(), input_test.cpu()
        output_grad_train, output_grad_test = output_grad_train.cpu(), output_grad_test.cpu()
    else:
        raise NotImplementedError("Unkown device type {}".format(device))

    def pad_input(x):
        padding_dim = (x.shape[0], x.shape[1], x.shape[-2] + padding*2, x.shape[-1] + padding*2)
        if device == "cuda":
            pad_input = torch.zeros(padding_dim).cuda()
        elif device == "cpu":
            pad_input = torch.zeros(padding_dim)
        else:
            raise NotImplementedError("Unkown device type {}".format(device))
        pad_input[:, :, padding: -padding, padding : -padding] = x

        return pad_input

    if "padding" in kwargs:
        padding = kwargs["padding"]
        input_train, input_test = pad_input(input_train), pad_input(input_test)

    kernel_size = kwargs["kernel_size"]

    _ = 0

    train_feats_dict, test_feats_dict = {}, {}
    train_outputs_grad_dict, test_outputs_grad_dict = {}, {}

    for m in range(kernel_size):
        for n in range(kernel_size):

            t1, t2 = output_grad_train.shape[-2], output_grad_train.shape[-1]
            _input_train_slice, _input_test_slice = input_train[:, :, m:m+t1, n:n+t2], input_test[:, :, m:m+t1, n:n+t2]

            if device == "cuda":
                _output_grad_train_slice, _output_grad_test_slice = torch.tensor(output_grad_train).cuda(), \
                                                                    torch.tensor(output_grad_test.cuda())
            elif device == "cpu":
                _output_grad_train_slice, _output_grad_test_slice = torch.tensor(output_grad_train), \
                                                                    torch.tensor(output_grad_test)
            else:
                raise NotImplementedError("Unkown device type {}".format(device))

            key = (m, n)
            train_feats_dict[key], test_feats_dict[key] = _input_train_slice, _input_test_slice
            train_outputs_grad_dict[key], test_outputs_grad_dict[key] = _output_grad_train_slice, _output_grad_test_slice

    meta_grad = _meta_grad_conv2d(train_feats_dict, train_outputs_grad_dict, test_feats_dict, test_outputs_grad_dict)

    return meta_grad

def batch_meta_grad_conv2d(input_train, test_feats_dict, output_grad_train, test_outputs_grad_dict, device, **kwargs):
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

        for name in test_feats_dict:
            test_feats_dict[name] = test_feats_dict[name].cuda()
        for name in test_outputs_grad_dict:
            test_outputs_grad_dict[name] = test_outputs_grad_dict[name].cuda()

    elif device == "cpu":
        input_train = input_train.cpu()
        output_grad_train = output_grad_train.cpu()

        for name in test_feats_dict:
            test_feats_dict[name] = test_feats_dict[name].cpu()
        for name in test_outputs_grad_dict:
            test_outputs_grad_dict[name] = test_outputs_grad_dict[name].cpu()

    else:
        raise NotImplementedError("Unkown device type {}".format(device))

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

    meta_grad = _meta_grad_conv2d(train_feats_dict, train_outputs_grad_dict, test_feats_dict, test_outputs_grad_dict)

    return meta_grad

def _meta_grad_conv2d(train_feats_dict, train_outputs_grad_dict, test_feats_dict, test_outputs_grad_dict):

    meta_grad = 0

    for key in train_feats_dict.keys():

        _input_train_slice, _input_test_slice = train_feats_dict[key], test_feats_dict[key].squeeze()
        _output_grad_train_slice, _output_grad_test_slice = train_outputs_grad_dict[key], test_outputs_grad_dict[key].squeeze()

        p1 = torch.tensordot(_input_train_slice, _input_test_slice, dims=([1], [0]))
        p2 = torch.tensordot(_output_grad_train_slice, _output_grad_test_slice, dims=([1], [0]))

        _sum_axis = tuple([i for i in range(1, len(p2.shape))])
        meta_grad += torch.multiply(p1, p2).sum(_sum_axis)

    return meta_grad



if __name__ == "__main__":
    #input channel is 1, output channel is 10, kernel size is 3
    input_train, input_test = torch.rand(size=(10, 1, 5, 5)), torch.rand(size=(1, 1, 5, 5))
    output_grad_train, output_grad_test = torch.rand(size=(10, 10, 5, 5)), torch.rand(size=(1, 10, 5, 5))
    args = {"kernel_size": 3, "padding": 1}

    meta_grad, grad_train_0, grad_test_0 = _test_meta_grad_conv2d(input_train, input_test, output_grad_train, output_grad_test, device="cpu", **args)

    ground_truth_meta_gradient = torch.multiply(grad_train_0, grad_test_0).sum()

    print (meta_grad[0], ground_truth_meta_gradient)

