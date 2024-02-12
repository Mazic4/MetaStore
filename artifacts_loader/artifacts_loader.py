import timeit

import torch

from ted_v3.utils.utils import *
from ted_v3.utils.utils import _merge_holders
from ted_v3.utils.grad_to_feats import *
from ted_v3.runtime_log import logger


import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _load_artifacts(artifacts_dict, artifacts_idx, artifacts_log_path_fn, artifacts_index_path, quant_mode):

    @timer(logger.io_time_per_layer)
    def _load_single_layer(artifacts_log_path_fn, name, quant_mode, **kwargs):
        artifacts = torch.load(artifacts_log_path_fn(name))
        if quant_mode == "quant":
            artifacts = torch.dequantize(artifacts)
        artifacts = artifacts.to(torch.float32)
        return artifacts

    for name in artifacts_dict:
        artifacts = _load_single_layer(artifacts_log_path_fn, name, quant_mode, log_name=name)
        artifacts_dict[name].append(artifacts)

    if artifacts_index_path is not None:
        artifacts_idx += torch.load(artifacts_index_path)

    return

def _concat_artifacts_gpu(artifacts_holder):

    @timer(logger.io_time_per_layer)
    def _concat_artifacts_single_layer(artifacts_list, **kwargs):
        artifacts_tensor = torch.cat(artifacts_list, dim=0).cuda()
        return artifacts_tensor

    for name in artifacts_holder:
        artifacts_tensor = _concat_artifacts_single_layer(artifacts_holder[name], log_name=name)
        artifacts_holder[name] = artifacts_tensor

    return


def load_data_artifacts_naive(analyzer, unvisited_batches):

    io_time_per_layer = {}

    analyzer.train_ind_grad_dict = {key: [] for key in analyzer.target_layers}
    analyzer.cached_batches = []
    analyzer.cached_train_samples_idx = []

    #gpu memory estimation
    total_file_size = 0
    for layer_name in analyzer.target_layers:
        total_file_size += analyzer.artifacts_size["naive"][layer_name]
    max_batch_size = analyzer.gpu_max_size / total_file_size

    # load batches
    while len(unvisited_batches) > 0 and len(analyzer.cached_batches) < max_batch_size:

        batch_idx = unvisited_batches.pop()

        artifacts_log_path_fn = lambda name: "{}/{}_batch_{}_ind_grad.pt".format(analyzer.artifacts_log_path,
                                                                                 name,
                                                                                 batch_idx)
        artifacts_index_path = "{}/naive_batch_{}_idx.pt".format(analyzer.artifacts_log_path, batch_idx)
        _load_artifacts(analyzer.train_ind_grad_dict,analyzer.cached_train_samples_idx,artifacts_log_path_fn,\
                        artifacts_index_path,analyzer.args.precision_mode)

        analyzer.cached_batches.append(batch_idx)

        _concat_artifacts_gpu(analyzer.train_ind_grad_dict)

    return unvisited_batches


def load_data_artifacts_metastore(analyzer, unvisited_batches):

    # clear cached artifacts of curr layers
    torch.cuda.empty_cache()

    analyzer.train_dy_dw_dict = {key: [] for key in analyzer.target_layers}
    analyzer.train_dl_dy_dict = {key: [] for key in analyzer.target_layers}
    analyzer.cached_batches = []
    analyzer.cached_train_samples_idx = []

    #gpu memory estimation
    total_file_size = 0
    for layer_name in analyzer.target_layers:
        total_file_size += analyzer.artifacts_size["ted_input_train"][layer_name]
        total_file_size += analyzer.artifacts_size["ted_output_grad_train"][layer_name]
    max_batch_size = analyzer.gpu_max_size / total_file_size

    # load batches
    while len(unvisited_batches) > 0 and len(analyzer.cached_batches) < max_batch_size:

        batch_idx = unvisited_batches.pop()

        artifacts_log_path_fn = lambda name: "{}/{}_batch_{}_dy_dw.pt".format(analyzer.artifacts_log_path,
                                                                              name,
                                                                              batch_idx)
        artifacts_index_path = "{}/ted_batch_{}_idx.pt".format(analyzer.artifacts_log_path, batch_idx)
        _load_artifacts(analyzer.train_dy_dw_dict,analyzer.cached_train_samples_idx,artifacts_log_path_fn,
                        artifacts_index_path,analyzer.args.precision_mode)

        artifacts_log_path_fn = lambda name: "{}/{}_batch_{}_dl_dy.pt".format(analyzer.artifacts_log_path,
                                                                              name,
                                                                              batch_idx)
        _load_artifacts(analyzer.train_dl_dy_dict,None,artifacts_log_path_fn,None,analyzer.args.precision_mode)
        analyzer.cached_batches.append(batch_idx)

    _concat_artifacts_gpu(analyzer.train_dy_dw_dict)
    _concat_artifacts_gpu(analyzer.train_dl_dy_dict)

    return unvisited_batches

def load_data_artifacts(analyzer, unvisited_batches, method="naive"):

    if method == "naive":
        return load_data_artifacts_naive(analyzer, unvisited_batches)
    elif method in ["ted", "half", "recon"]:
        return load_data_artifacts_metastore(analyzer, unvisited_batches)
    else:
        raise NotImplementedError

# def load_data_artifacts(self, unvisited_batches, method="naive"):
#     io_time_per_layer = {}
#
#     debug_timer = timeit.default_timer()
#
#     if method == "naive":
#
#         # clear cached artifacts of curr layers
#         self.train_ind_grad_dict = {key: [] for key in self.target_layers}
#         self.cached_batches = []
#         self.cached_train_samples_idx = []
#
#         # calculate how much memory is needed to cache one batch for all target layers
#         total_file_size = 0
#         for layer_name in self.target_layers:
#              total_file_size += self.artifacts_size["naive"][layer_name]
#         max_batch_size = self.gpu_max_size / total_file_size
#         max_batch_size = 2
#         if self.args.dataset == 'imagenet':
#             # max_batch_size = 2
#             max_batch_size = 1
#
#             # if self.args.method == "recon":
#             #     max_batch_size = min(1, max_batch_size)
#             # else:
#             #     max_batch_size = min(5, max_batch_size)
#         # load batches
#         while len(unvisited_batches) > 0 and len(self.cached_batches) < max_batch_size:
#
#             batch_idx = unvisited_batches.pop()
#
#             _train_data_idx = torch.load("{}/naive_batch_{}_idx.pt".format(self.artifacts_log_path, batch_idx))
#
#             self.cached_train_samples_idx += _train_data_idx
#             for layer_name in self.target_layers:
#                 s_time = timeit.default_timer()
#                 _train_ind_grad = torch.load("{}/{}_batch_{}_ind_grad.pt".
#                                              format(self.artifacts_log_path, layer_name, batch_idx))
#                 if self.args.precision_mode == "quant":
#                     _train_ind_grad = torch.dequantize(_train_ind_grad)
#                     _train_ind_grad = _train_ind_grad.to(torch.float32)
#                 else:
#                     _train_ind_grad = _train_ind_grad.to(torch.float32)
#
#                 self.train_ind_grad_dict[layer_name].append(_train_ind_grad)
#                 e_time = timeit.default_timer()
#                 io_time_per_layer[layer_name] = io_time_per_layer.get(layer_name, 0) + e_time - s_time
#
#             self.cached_batches.append(batch_idx)
#
#         self.cached_train_samples_idx = torch.tensor(self.cached_train_samples_idx)
#         for layer_name in self.train_ind_grad_dict:
#
#             s_time = timeit.default_timer()
#             train_ind_grad = torch.cat(self.train_ind_grad_dict[layer_name], dim=0)
#
#             if self.args.device == "cuda":
#                 train_ind_grad = train_ind_grad.cuda()
#             elif self.args.device == "cpu":
#                 train_ind_grad = train_ind_grad.cpu()
#             else:
#                 raise NotImplementedError("Unkown device type {}".format(self.args.device))
#
#             self.train_ind_grad_dict[layer_name] = train_ind_grad
#             e_time = timeit.default_timer()
#             io_time_per_layer[layer_name] += e_time - s_time
#
#     elif method in ["ted", "half", "recon"]:
#
#         # clear cached artifacts of curr layers
#         torch.cuda.empty_cache()
#         self.train_dy_dw_dict = {key: [] for key in self.target_layers}
#         self.train_dl_dy_dict = {key: [] for key in self.target_layers}
#         self.cached_batches = []
#         self.cached_train_samples_idx = []
#
#         # calculate how much memory is needed to cache one batch for all target layers
#         total_file_size = 0
#         for layer_name in self.target_layers:
#             total_file_size += self.artifacts_size["ted_input_train"][layer_name]
#             total_file_size += self.artifacts_size["ted_output_grad_train"][layer_name]
#         max_batch_size = self.gpu_max_size / total_file_size
#
#         if self.args.dataset == 'imagenet' and self.args.method in ["ted"]:
#             max_batch_size = min(20, max_batch_size)
#         elif self.args.dataset in ['imagenet', 'cifar10'] and method in ["half"]:
#             max_batch_size = min(5, max_batch_size)
#         elif self.args.dataset == 'imagenet' and self.args.method in ["iter", "recon"]:
#             total_file_size = 0
#             for layer_name in self.target_layers:
#                 total_file_size += self.artifacts_size["naive"][layer_name]
#             max_batch_size = self.gpu_max_size / total_file_size
#             # max_batch_size = min(20, max_batch_size)
#             max_batch_size = 5
#
#         # load batches
#         while len(unvisited_batches) > 0 and len(self.cached_batches) < max_batch_size:
#
#             batch_idx = unvisited_batches.pop()
#
#             _train_data_idx = torch.load("{}/ted_batch_{}_idx.pt".format(self.artifacts_log_path, batch_idx))
#             self.cached_train_samples_idx += _train_data_idx
#             self.cached_batches.append(batch_idx)
#
#             for layer_name in self.target_layers:
#                 s_time = timeit.default_timer()
#                 _train_dy_dw = torch.load("{}/{}_batch_{}_dy_dw.pt"
#                                           .format(self.artifacts_log_path, layer_name, batch_idx))
#                 _train_dl_dy = torch.load("{}/{}_batch_{}_dl_dy.pt"
#                                           .format(self.artifacts_log_path, layer_name, batch_idx))
#                 if self.args.precision_mode == "quant":
#                     _train_dy_dw = torch.dequantize(_train_dy_dw)
#                     _train_dl_dy = torch.dequantize(_train_dl_dy)
#                     _train_dy_dw = _train_dy_dw.to(torch.float32)
#                     _train_dl_dy = _train_dl_dy.to(torch.float32)
#                 else:
#                     _train_dy_dw = _train_dy_dw.to(torch.float32)
#                     _train_dl_dy = _train_dl_dy.to(torch.float32)
#
#                 self.train_dy_dw_dict[layer_name].append(_train_dy_dw)
#                 self.train_dl_dy_dict[layer_name].append(_train_dl_dy)
#                 e_time = timeit.default_timer()
#                 io_time_per_layer[layer_name] = io_time_per_layer.get(layer_name, 0) + e_time - s_time
#
#         self.cached_train_samples_idx = torch.tensor(self.cached_train_samples_idx)
#         for layer_name in self.target_layers:
#
#             s_time = timeit.default_timer()
#             train_dy_dw = torch.cat(self.train_dy_dw_dict[layer_name], dim=0)
#             train_dl_dy = torch.cat(self.train_dl_dy_dict[layer_name], dim=0)
#
#             if self.args.device == "cuda":
#                 train_dy_dw = train_dy_dw.cuda()
#                 train_dl_dy = train_dl_dy.cuda()
#             elif self.args.device == "cpu":
#                 train_dy_dw = train_dy_dw.cpu()
#                 train_dl_dy = train_dl_dy.cpu()
#             else:
#                 raise NotImplementedError("Unkown device type {}".format(self.args.device))
#
#             self.train_dy_dw_dict[layer_name] = train_dy_dw
#             self.train_dl_dy_dict[layer_name] = train_dl_dy
#             e_time = timeit.default_timer()
#             io_time_per_layer[layer_name] += e_time - s_time
#
#     else:
#         raise ValueError("pass")
#
#     return unvisited_batches, io_time_per_layer

