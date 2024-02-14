from backpack import backpack, extend
from backpack.extensions.firstorder.batch_grad import *

from utils.hooks import save_forward_hooks, save_backward_hooks
from utils.utils import *

from runtime_log import logger

def _register_hooks(analyzer):

    for name, layer in analyzer.flatten_model:
        if name not in analyzer.target_layers: continue
        # set the name of layer for saving intermediate results during bb
        layer.name = name
        layer.register_forward_hook(save_forward_hooks)
        layer.register_backward_hook(save_backward_hooks)

    return True

@timer(logger.preprocess_time)
def _run_backward_batch(analyzer, data, label, **kwargs):

    if analyzer.dataset_name in ["cifar10", "imagenet"]:

        model = extend(analyzer.model)
        loss_func = extend(analyzer.loss_func)

        if data.shape[0] == 1:
            label = torch.tensor(label).unsqueeze(0)

        data, label = Variable(data).cuda(), Variable(label).cuda()

        output_logits = model(data)

        loss = loss_func(output_logits, label)
        analyzer.optimizer.zero_grad()

        # --- extract all the individual gradients---
        batch_grad = BatchGrad()
        with backpack(batch_grad):
            loss.backward()

    elif analyzer.dataset_name == "AGNews":

        model = analyzer.model
        loss_func = analyzer.loss_func

        inputs, label = {key: to_var(value, False) for key, value in data.items()}, Variable(label).cuda()

        output_logits = model(**inputs).logits

        loss = loss_func(output_logits, label)
        analyzer.optimizer.zero_grad()

        loss.backward()

    else:
        raise ValueError("Unknown Dataset {}.".format(analyzer.dataset_name))

    return True


def _collect_artifacts_naive(analyzer):

    artifacts_holder = {}

    for name, layer in analyzer.flatten_model:
        param = [params for params in layer.parameters()][0]
        # example: _name: l1.weight, name: l1
        if name not in analyzer.target_layers: continue
        if len(param.grad.shape) == 1: continue

        if analyzer.dataset_name in ["cifar10", "imagenet"]:
            artifacts_holder[name] = param.grad_batch
        elif analyzer.dataset_name == "AGNews":
            artifacts_holder[name] = param.grad
        else:
            raise ValueError("Unknown Dataset {}.".format(analyzer.dataset_name))

    return artifacts_holder


def _collect_artifacts_metastore(analyzer):

    artifacts_holder_1 = {}
    artifacts_holder_2 = {}

    for name, layer in analyzer.flatten_model:
        if name not in analyzer.target_layers: continue
        dy_dw = torch.load("{}/_{}_input.pt".format(analyzer._artifacts_log_path, name))
        dl_dy = torch.load("{}/_{}_output_grad.pt".format(analyzer._artifacts_log_path, name))
        artifacts_holder_1[name] = torch.tensor(dl_dy).cuda()
        artifacts_holder_2[name] = torch.tensor(dy_dw).cuda()

    return artifacts_holder_1, artifacts_holder_2


def _get_data_artifacts_naive(analyzer, data, label):

    try:
         _run_backward_batch(analyzer, data, label, log_name="backward")
    except Exception as e:
        print(e)

    artifacts_holder = _collect_artifacts_naive(analyzer)

    return [artifacts_holder]

def _get_data_artifacts_metastore(analyzer, data, label):

    _register_hooks(analyzer)

    try:
        _run_backward_batch(analyzer, data, label, log_name="backward")
    except Exception as e:
        print(e)

    artifacts_holder_1, artifacts_holder_2 = _collect_artifacts_metastore(analyzer)

    return  [artifacts_holder_1, artifacts_holder_2]


def _get_data_artifacts(analyzer, data, label, method="naive"):
    # calculat the data artifacts during backpropagation

    if method in ["naive"]:

        artifacts_holder_list = _get_data_artifacts_naive(analyzer, data, label)

        return artifacts_holder_list

    elif method in ["ted", "half", "recon"]:

        artifacts_holder_list = _get_data_artifacts_metastore(analyzer, data, label)

        return artifacts_holder_list

    else:
        raise ValueError("The method {} is not implemented yet.".format(method))