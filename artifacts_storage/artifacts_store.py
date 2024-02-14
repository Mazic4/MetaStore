import copy

from utils.hooks import save_forward_hooks, save_backward_hooks
from artifacts_calculator.artifacts_collector import _get_data_artifacts

from utils.utils import *
from utils.utils import _merge_holders

from runtime_log import logger

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
        if quant_mode:
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

        if analyzer.dataset_name in ["cifar10", "imagenet"]:

            artifacts_holder = _get_data_artifacts(analyzer, data, label, method="naive")[0]

            artifacts_cache_loader = _merge_holders(artifacts_holder, artifacts_cache_loader)
            artifacts_index_loader += idx.tolist()
            print (batch_idx, artifacts_index_loader)
            print (len(analyzer.train_loader))

            if len(artifacts_index_loader) >= analyzer.max_store_batch_size or \
                    batch_idx == len(analyzer.train_loader)-1:

                artifacts_log_path_fn = lambda name: "{}/{}_batch_{}_ind_grad.pt".format(analyzer.artifacts_log_path,
                                                                                         name,
                                                                                         _cache_idx)
                artifacts_index_log_path = "{}/naive_batch_{}_idx.pt".format(analyzer.artifacts_log_path, _cache_idx)
                _store_artifacts(artifacts_cache_loader,artifacts_index_loader,artifacts_log_path_fn,
                                 artifacts_index_log_path,quant_mode = analyzer.use_quant,
                                 quant_type=analyzer.default_quant_type)

                _cache_idx += 1
                artifacts_cache_loader = {}
                artifacts_index_loader = []

        elif analyzer.dataset_name == "AGNews":

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
                                                                                     batch_idx)
            artifacts_index_log_path = "{}/naive_batch_{}_idx.pt".format(analyzer.artifacts_log_path, batch_idx)
            _store_artifacts(artifacts_cache_loader,batch_indices,artifacts_log_path_fn,artifacts_index_log_path,
                             quant_mode=analyzer.use_quant,quant_type=analyzer.default_quant_type)

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
        if len(artifacts_index_loader) >= analyzer.max_store_batch_size or \
                batch_idx == len(analyzer.train_loader)-1:

            artifacts_log_path_fn = lambda name: "{}/{}_batch_{}_dy_dw.pt".format(analyzer.artifacts_log_path,
                                                                                  name,
                                                                                  _cache_idx)
            artifacts_index_log_path = "{}/ted_batch_{}_idx.pt".format(analyzer.artifacts_log_path, _cache_idx)
            _store_artifacts(artifacts_cache_loader_dy_dw,artifacts_index_loader,artifacts_log_path_fn,
                             artifacts_index_log_path,quant_mode=analyzer.use_quant,
                             quant_type=analyzer.default_quant_type)

            artifacts_log_path_fn = lambda name:"{}/{}_batch_{}_dl_dy.pt".format(analyzer.artifacts_log_path,
                                                                                 name,
                                                                                 _cache_idx)
            _store_artifacts(artifacts_cache_loader_dl_dy,None,artifacts_log_path_fn,None,
                             quant_mode=analyzer.use_quant,quant_type=analyzer.default_quant_type)

            _cache_idx += 1
            artifacts_cache_loader_dy_dw = {}
            artifacts_cache_loader_dl_dy = {}
            artifacts_index_loader = []

    return


def store_traindata_artifacts(analyzer, method="naive"):

    if method in ["naive"]:
        store_traindata_artifacts_naive(analyzer)
    elif method in ["ted", "half", "recon"]:
        store_traindata_artifacts_metastore(analyzer)
    else:
        raise NotImplementedError

    return

