import copy
import os
import timeit

from ted_v3.utils.hooks import save_forward_hooks, save_backward_hooks
from ted_v3.artifacts_calculator.artifacts_collector import _get_data_artifacts

from ted_v3.utils.utils import *
from ted_v3.utils.utils import _merge_holders
from ted_v3.utils.grad_to_feats import *

from ted_v3.runtime_log import logger

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _store_artifacts(artifacts_holder,
                     index_holder,
                     artifacts_log_path_fn,
                     artifacts_index_log_path,
                     quant_mode = "normal",
                     quant_type = None):

    @timer(logger.preprocess_time)
    def _store_artifacts_single_layer(artifacts_log_path_fn, name, quant_mode, quant_type, **kwargs):
        if quant_mode == "quant":
            artifacts_holder[name] = torch.quantize_per_tensor(artifacts_holder[name], 100, 0, quant_type)
        torch.save(artifacts_holder[name].cpu(), artifacts_log_path_fn(name))
        return

    for name in artifacts_holder:
        _store_artifacts_single_layer(artifacts_log_path_fn, name, quant_mode, quant_type, log_name=name)

    if index_holder is not None:
        torch.save(index_holder, artifacts_index_log_path)

    return


def store_traindata_artifacts_naive(analyzer):

    _cache_idx = 0
    artifacts_cache_loader = {}
    artifacts_index_loader = []

    for batch_idx, (idx, data, label) in enumerate(analyzer.train_loader):

        if analyzer.args.dataset in ["cifar10", "imagenet"]:

            artifacts_holder = _get_data_artifacts(analyzer, data, label, method="naive")[0]

            artifacts_cache_loader = _merge_holders(artifacts_holder, artifacts_cache_loader)
            artifacts_index_loader += idx.tolist()

            if len(artifacts_index_loader) >= analyzer.args.max_store_batch_size:

                artifacts_log_path_fn = lambda name: "{}/{}_batch_{}_ind_grad.pt".format(analyzer.artifacts_log_path,
                                                                                         name,
                                                                                         _cache_idx)
                artifacts_index_log_path = "{}/naive_batch_{}_idx.pt".format(analyzer.artifacts_log_path, _cache_idx)
                _store_artifacts(artifacts_cache_loader,artifacts_index_loader,artifacts_log_path_fn,
                                 artifacts_index_log_path,quant_mode = analyzer.args.precision_mode,
                                 quant_type=analyzer.default_quant_type)

                _cache_idx += 1
                artifacts_cache_loader = {}
                artifacts_index_loader = []

        elif analyzer.args.dataset == "AGNews":

            batch_indices = []
            batch_grad = {}
            data = {key: to_var(value, False) for key, value in data.items()}

            #for transformers, backpack doesn't support batch calculating
            for _ in range(len(idx)):

                _data = {
                    "input_ids": data["input_ids"][_].unsqueeze(0),
                    "attention_mask": data["attention_mask"][_].unsqueeze(0)
                }
                _label = Variable(label[_]).cuda().unsqueeze(0)
                _idx = idx[_]

                artifacts_holder = _get_data_artifacts(analyzer, _data, _label, method="naive")[0]

                for name in artifacts_holder:
                    if name not in batch_grad: batch_grad[name] = []
                    batch_grad[name].append(copy.deepcopy(artifacts_holder[name].unsqueeze(0)))
                batch_indices += [_idx]

            batch_indices = torch.tensor(batch_indices)
            for name in batch_grad:
                artifacts_cache_loader[name] = torch.cat(batch_grad[name], dim=0)

            #save grad
            artifacts_log_path_fn = lambda name: "{}/{}_batch_{}_ind_grad.pt".format(analyzer.artifacts_log_path,
                                                                                     name,
                                                                                     _cache_idx)
            artifacts_index_log_path = "{}/naive_batch_{}_idx.pt".format(analyzer.artifacts_log_path, batch_idx)
            _store_artifacts(artifacts_cache_loader,batch_indices,artifacts_log_path_fn,artifacts_index_log_path,
                             quant_mode=analyzer.args.precision_mode,quant_type=analyzer.default_quant_type)

        return

def _register_hooks(analyzer):

    for name, layer in analyzer.flatten_model:
        if name not in analyzer.target_layers: continue
        # set the name of layer for saving intermediate results during bb
        layer.name = name
        layer.register_forward_hook(save_forward_hooks)
        layer.register_backward_hook(save_backward_hooks)

    return True


def store_traindata_artifacts_metastore(analyzer):

    _register_hooks(analyzer)

    _cache_idx = 0
    artifacts_cache_loader_dy_dw = {}
    artifacts_cache_loader_dl_dy = {}
    artifacts_index_loader = []

    # here we should extract all the intermediate features and individual gradients of training data
    for batch_idx, (idx, data, label) in enumerate(analyzer.train_loader):

        artifacts_holder_list = _get_data_artifacts(analyzer,data,label,method="ted")
        artifacts_holder_dl_dy, artifacts_holder_dy_dw = artifacts_holder_list[0], artifacts_holder_list[1]

        artifacts_cache_loader_dy_dw = _merge_holders(artifacts_holder_dy_dw, artifacts_cache_loader_dy_dw)
        artifacts_cache_loader_dl_dy = _merge_holders(artifacts_holder_dl_dy, artifacts_cache_loader_dl_dy)
        artifacts_index_loader += idx.tolist()

        # print (timeit.default_timer()-start_time)
        if len(artifacts_index_loader) >= analyzer.args.max_store_batch_size:

            artifacts_log_path_fn = lambda name: "{}/{}_batch_{}_dy_dw.pt".format(analyzer.artifacts_log_path,
                                                                                  name,
                                                                                  _cache_idx)
            artifacts_index_log_path = "{}/ted_batch_{}_idx.pt".format(analyzer.artifacts_log_path, _cache_idx)
            _store_artifacts(artifacts_cache_loader_dy_dw,artifacts_index_loader,artifacts_log_path_fn,
                             artifacts_index_log_path,quant_mode=analyzer.args.precision_mode,
                             quant_type=analyzer.default_quant_type)

            artifacts_log_path_fn = lambda name:"{}/{}_batch_{}_dl_dy.pt".format(analyzer.artifacts_log_path,
                                                                                 name,
                                                                                 _cache_idx)
            _store_artifacts(artifacts_cache_loader_dl_dy,None,artifacts_log_path_fn,None,
                             quant_mode=analyzer.args.precision_mode,quant_type=analyzer.default_quant_type)

            _cache_idx += 1
            artifacts_cache_loader_dy_dw = {}
            artifacts_cache_loader_dl_dy = {}
            artifacts_index_loader = []

    return

# def store_traindata_artifacts(self, method="naive", debug=False) -> dict:
#     # A controller to collect artifacts of training data
#
#     max_store_batch_size = self.args.max_store_batch_size
#
#     start_time = timeit.default_timer()
#
#     save_time_per_layer = {}
#     p_n = self.args.float_precision
#
#     if method in ["naive"]:
#
#         total_backward_time = 0
#         _cache_idx = 0
#         artifacts_cache_loader = {}
#         artifacts_index_loader = []
#
#         for batch_idx, (idx, data, label) in enumerate(self.train_loader):
#
#             # print ("batch", batch_idx)
#
#             if self.args.dataset in ["cifar10", "imagenet"]:
#                 artifacts_holder, backward_time = _get_data_artifacts(self, data, label, method="naive")
#
#                 total_backward_time += backward_time
#
#                 for name in artifacts_holder:
#                     # print (artifacts_holder[name].to(self.default_dtype).cpu() * 10000)
#
#                     if name not in artifacts_cache_loader or len(artifacts_cache_loader[name]) == 0:
#                         artifacts_cache_loader[name] = artifacts_holder[name]
#                     else:
#                         artifacts_cache_loader[name] = torch.cat([artifacts_cache_loader[name], artifacts_holder[name]], axis=0)
#
#                 artifacts_index_loader += idx.tolist()
#
#                 if len(artifacts_index_loader) >= max_store_batch_size:
#                     for name in artifacts_cache_loader:
#                         s_time = timeit.default_timer()
#
#                         if self.args.precision_mode == "quant":
#                             artifacts_cache_loader[name] = torch.quantize_per_tensor(artifacts_cache_loader[name], 100, 0,
#                                                                                self.default_quant_type)
#                         torch.save(artifacts_cache_loader[name].cpu(),
#                                    "{}/{}_batch_{}_ind_grad.pt".format(self.artifacts_log_path, name, _cache_idx))
#                         save_time_per_layer[name] = save_time_per_layer.get(name, 0) + timeit.default_timer() - s_time
#
#                     torch.save(artifacts_index_loader, "{}/naive_batch_{}_idx.pt".format(self.artifacts_log_path, _cache_idx))
#
#                     _cache_idx += 1
#                     artifacts_cache_loader = {}
#                     artifacts_index_loader = []
#
#             elif self.args.dataset == "AGNews":
#                 batch_indices = []
#                 batch_grad = {}
#                 data = {key: to_var(value, False) for key, value in data.items()}
#                 for _ in range(len(idx)):
#
#                     _data = {"input_ids": data["input_ids"][_].unsqueeze(0),
#                              "attention_mask": data["attention_mask"][_].unsqueeze(0)}
#                     _label = Variable(label[_]).cuda().unsqueeze(0)
#                     _idx = idx[_]
#
#                     artifacts_holder, backward_time = _get_data_artifacts(self, _data, _label, method="naive")
#                     total_backward_time += backward_time
#                     for name in artifacts_holder:
#                         if name not in batch_grad: batch_grad[name] = []
#                         batch_grad[name].append(copy.deepcopy(artifacts_holder[name].unsqueeze(0)))
#                     batch_indices += [_idx]
#                 batch_indices = torch.tensor(batch_indices)
#                 for name in batch_grad:
#                     batch_grad[name] = torch.cat(batch_grad[name], dim=0)
#
#                 for name in batch_grad:
#                     s_time = timeit.default_timer()
#                     # Save the average gradient. Thus each gradient must normalize by the batch size.
#                     if self.args.precision_mode == "quant":
#                         batch_grad[name] = torch.quantize_per_tensor(batch_grad[name], 100, 0,
#                                                                            self.default_quant_type)
#                     torch.save(batch_grad[name].cpu() / len(batch_indices),
#                                "{}/{}_batch_{}_ind_grad.pt".format(self.artifacts_log_path, name, batch_idx))
#                     save_time_per_layer[name] = save_time_per_layer.get(name, 0) + timeit.default_timer() - s_time
#
#                 torch.save(batch_indices, "{}/naive_batch_{}_idx.pt".format(self.artifacts_log_path, batch_idx))
#
#
#     elif method in ["ted", "half", "recon"]:
#
#         if debug:
#             print("These layers are hooked.")
#             print("------------------------------")
#
#         for name, layer in self.flatten_model:
#             print (name)
#             if name in self.target_layers:
#                 # set the name of layer for saving intermediate results during bb
#                 layer.name = name
#                 layer.register_forward_hook(save_forward_hooks)
#                 layer.register_backward_hook(save_backward_hooks)
#
#                 if debug:
#                     print(layer)
#
#         if debug:
#             print("------------------------------")
#
#         _cache_idx = 0
#         artifacts_cache_loader_dy_dw = {}
#         artifacts_cache_loader_dl_dy = {}
#         artifacts_index_loader = []
#
#         total_backward_time = 0
#         # here we should extract all the intermediate features and individual gradients of training data
#         for batch_idx, (idx, data, label) in enumerate(self.train_loader):
#
#             # print ("batch", batch_idx)
#
#             artifacts_holder_dl_dy, artifacts_holder_dy_dw, backward_time = _get_data_artifacts(self, data, label,
#                                                                                                      method="ted")
#
#             total_backward_time += backward_time
#
#             for name in artifacts_holder_dy_dw:
#                 s_time = timeit.default_timer()
#                 if name not in artifacts_cache_loader_dy_dw or len(artifacts_cache_loader_dy_dw[name]) == 0:
#                     artifacts_cache_loader_dy_dw[name] = artifacts_holder_dy_dw[name]
#                 else:
#                     artifacts_cache_loader_dy_dw[name] = torch.cat([artifacts_cache_loader_dy_dw[name],
#                                                                     artifacts_holder_dy_dw[name]],
#                                                                    axis=0)
#
#             for name in artifacts_holder_dl_dy:
#                 s_time = timeit.default_timer()
#                 if name not in artifacts_cache_loader_dl_dy or len(artifacts_cache_loader_dl_dy[name]) == 0:
#                     artifacts_cache_loader_dl_dy[name] = artifacts_holder_dl_dy[name]
#                 else:
#                     artifacts_cache_loader_dl_dy[name] = torch.cat([artifacts_cache_loader_dl_dy[name],
#                                                                     artifacts_holder_dl_dy[name]],
#                                                                    axis=0)
#
#             artifacts_index_loader += idx.tolist()
#
#             # print (timeit.default_timer()-start_time)
#             if len(artifacts_index_loader) >= max_store_batch_size:
#                 for name in artifacts_cache_loader_dy_dw:
#                     s_time = timeit.default_timer()
#                     if self.args.precision_mode == "quant":
#                         artifacts_cache_loader_dy_dw[name] = torch.quantize_per_tensor(artifacts_cache_loader_dy_dw[name], 100, 0,
#                                                                                  self.default_quant_type)
#                         artifacts_cache_loader_dl_dy[name] = torch.quantize_per_tensor(artifacts_cache_loader_dl_dy[name], 100, 0,
#                                                                                  self.default_quant_type)
#                     torch.save(artifacts_cache_loader_dy_dw[name].cpu(),
#                                "{}/{}_batch_{}_dy_dw.pt".format(self.artifacts_log_path, name, _cache_idx))
#                     torch.save(artifacts_cache_loader_dl_dy[name].cpu(),
#                                "{}/{}_batch_{}_dl_dy.pt".format(self.artifacts_log_path, name, _cache_idx))
#
#                     save_time_per_layer[name] = save_time_per_layer.get(name, 0) + timeit.default_timer() - s_time
#
#                 torch.save(artifacts_index_loader, "{}/ted_batch_{}_idx.pt".format(self.artifacts_log_path, _cache_idx))
#
#                 _cache_idx += 1
#                 artifacts_cache_loader_dy_dw = {}
#                 artifacts_cache_loader_dl_dy = {}
#                 artifacts_index_loader = []
#
#     else:
#         raise NotImplementedError("The method {} is not implemented yet.".format(method))
#
#     print("The total backward time of {} is:{} ".format(method, total_backward_time))
#
#     return total_backward_time, save_time_per_layer


def store_traindata_artifacts(analyzer, method="naive"):

    if method in ["naive"]:
        store_traindata_artifacts_naive(analyzer)
    elif method in ["ted", "half", "recon"]:
        store_traindata_artifacts_metastore(analyzer)
    else:
        raise NotImplementedError

    return

