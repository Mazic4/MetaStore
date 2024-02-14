from utils.utils import *
from utils.grad_to_feats import *
from runtime_log import logger

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _load_artifacts(artifacts_dict, artifacts_idx, artifacts_log_path_fn, artifacts_index_path, quant_mode):

    @timer(logger.io_time_per_layer)
    def _load_single_layer(artifacts_log_path_fn, name, quant_mode, **kwargs):
        artifacts = torch.load(artifacts_log_path_fn(name))
        if quant_mode:
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
        total_file_size += analyzer.artifacts_size[layer_name]["naive"]
    max_batch_size = analyzer.gpu_max_size / total_file_size

    # load batches
    while len(unvisited_batches) > 0 and len(analyzer.cached_batches) < max_batch_size:

        batch_idx = unvisited_batches.pop()

        artifacts_log_path_fn = lambda name: "{}/{}_batch_{}_ind_grad.pt".format(analyzer.artifacts_log_path,
                                                                                 name,
                                                                                 batch_idx)
        artifacts_index_path = "{}/naive_batch_{}_idx.pt".format(analyzer.artifacts_log_path, batch_idx)
        _load_artifacts(analyzer.train_ind_grad_dict,analyzer.cached_train_samples_idx,artifacts_log_path_fn,\
                        artifacts_index_path,analyzer.use_quant)

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
        total_file_size += analyzer.artifacts_size[layer_name]["ted_input_train"]
        total_file_size += analyzer.artifacts_size[layer_name]["ted_output_grad_train"]
    max_batch_size = analyzer.gpu_max_size / total_file_size

    # load batches
    while len(unvisited_batches) > 0 and len(analyzer.cached_batches) < max_batch_size:

        batch_idx = unvisited_batches.pop()

        artifacts_log_path_fn = lambda name: "{}/{}_batch_{}_dy_dw.pt".format(analyzer.artifacts_log_path,
                                                                              name,
                                                                              batch_idx)
        artifacts_index_path = "{}/ted_batch_{}_idx.pt".format(analyzer.artifacts_log_path, batch_idx)
        _load_artifacts(analyzer.train_dy_dw_dict,analyzer.cached_train_samples_idx,artifacts_log_path_fn,
                        artifacts_index_path,analyzer.use_quant)

        artifacts_log_path_fn = lambda name: "{}/{}_batch_{}_dl_dy.pt".format(analyzer.artifacts_log_path,
                                                                              name,
                                                                              batch_idx)
        _load_artifacts(analyzer.train_dl_dy_dict,None,artifacts_log_path_fn,None,analyzer.use_quant)
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
