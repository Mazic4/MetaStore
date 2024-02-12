import torch.nn as nn


def unwrap(model: nn.Module, name: str, valid_layer: list):

    children = list(model.named_children())
    if len(children) == 0:
        for i in valid_layer:
            if isinstance(model, i):
                return [(name, model)]
        return []
    flatten_children = []
    for layer_name, layer in children:
        flatten_children.extend(
            unwrap(layer, name + "_" + layer_name, valid_layer))
    return flatten_children


if __name__ == '__main__':

    from ..model_trainers.models import *

    model = Network1()
    flatten_model = unwrap(model, "net", [nn.Linear, nn.Conv2d])
    for name, layer in flatten_model:
        layer.name = name
        layer.register_forward_hook(save_forward_hooks)
        layer.register_backward_hook(save_backward_hooks)
    x = tensor([1., 2.])
    y = tensor([2., 1.])
    output = model.forward(x)
    loss = F.mse_loss(output, y)
    loss.backward()
