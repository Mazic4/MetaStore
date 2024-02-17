import torch.nn as nn

from artifacts_loader.artifacts_loader import load_data_artifacts

import core_engine.ted_layers as ted_layers
import core_engine.half_layers as half_layers
import core_engine.recon_layers as recon_layers

from utils.utils import *

from runtime_log import logger


import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def _parse_test_artifacts(artifacts_list, method):
    if method in ["naive", "recon", "half"]:
        artifacts_dict = {
            "test_ind_grad_dict": artifacts_list[0]
        }
    elif method in ["ted"]:
        artifacts_dict = {
            "test_dl_dy_dict": artifacts_list[0],
            "test_dy_dw_dict": artifacts_list[1],
        }
    else:
        raise NotImplementedError
    return artifacts_dict

def _parse_layer_type(analyzer):
    layer_type_dict = {}
    for layer_name, layer in analyzer.flatten_model:
        if isinstance(layer, nn.Linear):
            layer_type_dict[layer_name] = "linear"
        elif isinstance(layer, nn.Conv2d):
            layer_type_dict[layer_name] = "conv"
        else:
            raise ValueError
    return layer_type_dict

def _get_meta_gradient_curr_batch_naive(train_artifacts_dict, test_artifacts_dict):

    @timer(logger.cal_time_per_layer)
    def _cal_meta_gradient_single_layer(train_artifacts, test_artifacts, **kwargs):
        _sum_axis = tuple([i for i in range(1, len(train_artifacts.shape))])
        meta_gradients = torch.multiply(test_artifacts, train_artifacts).sum(axis=_sum_axis)
        return meta_gradients

    meta_gradients = 0

    for name in test_artifacts_dict:
        train_artifacts = train_artifacts_dict[name]
        test_artifacts = test_artifacts_dict[name]
        meta_gradients_per_layer = _cal_meta_gradient_single_layer(train_artifacts, test_artifacts, log_name=name)
        meta_gradients += meta_gradients_per_layer

    return meta_gradients

def _get_meta_gradient_curr_batch_ted(train_dy_dw_dict,train_dl_dy_dict,test_dy_dw_dict,test_dl_dy_dict,layer_type_dict):

    @timer(logger.cal_time_per_layer)
    def _cal_meta_gradient_single_layer(train_dy_dw, train_dl_dy, test_dy_dw, test_dl_dy, layer_type, **kwargs):

        _fnc_dict = {
            "linear": ted_layers.meta_grad_linear,
            "conv":ted_layers.meta_grad_conv2d,
        }

        _kwgs_dict = {
            "linear": {"device":"cuda"},
            "conv": {"device":"cuda", "kernel_size": 3, "padding": 1}
        }

        return _fnc_dict[layer_type](train_dy_dw, test_dy_dw, train_dl_dy, test_dl_dy, **_kwgs_dict[layer_type])

    meta_gradients = 0

    for name in test_dy_dw_dict:
        test_dy_dw, test_dl_dy = test_dy_dw_dict[name], test_dl_dy_dict[name]
        train_dy_dw, train_dl_dy = train_dy_dw_dict[name], train_dl_dy_dict[name]
        meta_gradients_per_layer = _cal_meta_gradient_single_layer(train_dy_dw,
                                                                   train_dl_dy,
                                                                   test_dy_dw,
                                                                   test_dl_dy,
                                                                   layer_type_dict[name],
                                                                   log_name=name)
        meta_gradients += meta_gradients_per_layer

    return meta_gradients

def _get_meta_gradient_curr_batch_half(train_dy_dw_dict,train_dl_dy_dict,test_ind_grad_dict,layer_type_dict):
    @timer(logger.cal_time_per_layer)
    def _cal_meta_gradient_single_layer(train_dy_dw,train_dl_dy,test_ind_grad,layer_type, **kwargs):
        _fnc_dict = {
            "linear": half_layers.half_bert_meta_grad_linear,
            "conv": half_layers.half_meta_grad_conv2d,
        }

        _kwgs_dict = {
            "linear": {"device": "cuda"},
            "conv": {"device": "cuda", "kernel_size": 3, "padding": 1}
        }

        return _fnc_dict[layer_type](train_dy_dw,train_dl_dy,test_ind_grad, **_kwgs_dict[layer_type])

    meta_gradients = 0

    for name in test_ind_grad_dict:

        test_ind_grad = test_ind_grad_dict[name]
        train_dy_dw, train_dl_dy = train_dy_dw_dict[name], train_dl_dy_dict[name]

        meta_gradients_per_layer = _cal_meta_gradient_single_layer(train_dy_dw,
                                                                   train_dl_dy,
                                                                   test_ind_grad,
                                                                   layer_type_dict[name],
                                                                   log_name=name)
        meta_gradients += meta_gradients_per_layer

    return meta_gradients

def _get_meta_gradient_curr_batch_recon(train_dy_dw_dict,train_dl_dy_dict,test_ind_grad_dict,layer_type_dict):

    @timer(logger.cal_time_per_layer)
    def _cal_meta_gradient_single_layer(train_dy_dw, train_dl_dy, test_ind_grad, layer_type, **kwargs):
        _fnc_dict = {
            "linear": recon_layers.recon_meta_grad_linear,
            "conv": recon_layers.recon_meta_grad_conv2d,
        }

        _kwgs_dict = {
            "linear": {"device": "cuda"},
            "conv": {"device": "cuda", "kernel_size": 3, "padding": 1}
        }

        return _fnc_dict[layer_type](train_dy_dw, train_dl_dy, test_ind_grad, **_kwgs_dict[layer_type])

    meta_gradients = 0

    for name in test_ind_grad_dict:
        test_ind_grad = test_ind_grad_dict[name]
        train_dy_dw, train_dl_dy = train_dy_dw_dict[name], train_dl_dy_dict[name]

        meta_gradients_per_layer = _cal_meta_gradient_single_layer(train_dy_dw,
                                                                   train_dl_dy,
                                                                   test_ind_grad,
                                                                   layer_type_dict[name],
                                                                   log_name=name)
        meta_gradients += meta_gradients_per_layer

    return meta_gradients


def _get_meta_gradients_curr_batch(analyzer, method, **kwargs):

    if method == "naive":
        test_ind_grad_dict = kwargs['test_ind_grad_dict']
        train_ind_grad_dict = analyzer.train_ind_grad_dict
        meta_gradients = _get_meta_gradient_curr_batch_naive(train_ind_grad_dict,test_ind_grad_dict)

    elif method == "ted":
        test_dy_dw_dict, test_dl_dy_dict = kwargs['test_dy_dw_dict'],kwargs['test_dl_dy_dict']
        train_dy_dw_dict,train_dl_dy_dict = analyzer.train_dy_dw_dict, analyzer.train_dl_dy_dict

        layer_type_dict = _parse_layer_type(analyzer)

        meta_gradients = _get_meta_gradient_curr_batch_ted(train_dy_dw_dict,
                                                           train_dl_dy_dict,
                                                           test_dy_dw_dict,
                                                           test_dl_dy_dict,
                                                           layer_type_dict)
    elif method == "half":

        test_ind_grad_dict = kwargs['test_ind_grad_dict']
        train_dy_dw_dict, train_dl_dy_dict = analyzer.train_dy_dw_dict, analyzer.train_dl_dy_dict

        layer_type_dict = _parse_layer_type(analyzer)
        meta_gradients = _get_meta_gradient_curr_batch_half(train_dy_dw_dict,
                                                            train_dl_dy_dict,
                                                            test_ind_grad_dict,
                                                            layer_type_dict)

    elif method == "recon":

        test_ind_grad_dict = kwargs['test_ind_grad_dict']
        train_dy_dw_dict, train_dl_dy_dict = analyzer.train_dy_dw_dict, analyzer.train_dl_dy_dict

        layer_type_dict = _parse_layer_type(analyzer)
        meta_gradients = _get_meta_gradient_curr_batch_recon(train_dy_dw_dict,
                                                             train_dl_dy_dict,
                                                             test_ind_grad_dict,
                                                             layer_type_dict)

    else:
        raise NotImplementedError


    return meta_gradients


@timer(logger.end2end_query_time)
def _get_meta_gradients(analyzer, artifacts_list, method, **kwargs):
    # calcualte the meta-gradients based on the artifacts or ind gradinet

    meta_gradients = torch.zeros(analyzer.args["data"]["num_analyzed_samples"]).cuda()

    # find the unvisited_batches
    unvisited_batches = []
    for i in range(analyzer.num_batches):
        if i not in analyzer.cached_batches:
            unvisited_batches.append(i)

    # iterat through unvisited batches until all batches are visited
    while True:
        if len(analyzer.cached_batches) == 0 and method in ["ted", "naive", "recon", "half"]:
            #todo: rename this 'load_data_artifacts' to 'load_batch_artifacts
            unvisited_batches = load_data_artifacts(analyzer, unvisited_batches, method)

        artifacts_dict = _parse_test_artifacts(artifacts_list, method=method)
        _meta_gradients = _get_meta_gradients_curr_batch(analyzer=analyzer,method=method,**artifacts_dict)
        meta_gradients[analyzer.cached_train_samples_idx] = _meta_gradients

        #release the cached batches
        analyzer.cached_batches = []
        torch.cuda.empty_cache()

        if len(unvisited_batches) == 0:
            break

    return meta_gradients