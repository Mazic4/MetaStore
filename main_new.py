import argparse
import os.path
import random
import pickle
import sys

sys.path.append('../')
print(sys.path)

import torch.nn as nn
import torch.optim as optim

from model_trainers.models import VGG16, Customized_Bert_Model, ResNet50
from utils.data_loader_new import *
from utils.utils import *
from utils.grad_to_feats import *
from _analyzer_new import Analyzer
from artifacts_calculator.artifacts_collector import _get_data_artifacts
from artifacts_storage.artifacts_store import store_traindata_artifacts
from query.query_executor import query
from query.query_batch_executor import query_batch

from runtime_log import logger

import yaml

def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default="ted", help="method")
    parser.add_argument("--target_model", type=str, default="Bert", help="model")
    parser.add_argument("--data", type=str, default="AGNews", help="data")
    parser.add_argument("--num_analyzed_samples", type=int, default=100, help="num of analyzed data")

    return parser.parse_args()

def merge_configs(yaml_config, cmd_args):

    for key, value in vars(cmd_args).items():
        if key in ["target_model", "data"]:
            #exact config
            new_config = yaml_config[key].get(value, None)
            if new_config is not None:
                yaml_config[key] = new_config
            else:
                raise ValueError("Cannot find value {} with key {} in yaml.".format(value, key))

        if key == "num_analyzed_samples":
            yaml_config["data"][key] = value

    return yaml_config

def test_merge_config():
    config = read_yaml_config("./config.yaml")
    parse = parse_args()
    final_config = merge_configs(config, parse)
    return final_config

def main(opt):
    # TODO: remove this, and create a test script.

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    print("Set seed :", seed)

    analyzer = Analyzer(opt)

    if opt["data"]["data_name"] == "cifar10":
        test_loader, test_dataset = get_dataloader_cifar(opt["data"], mode="test", indices=range(10))
    else:
        test_loader, test_dataset = get_dataloader_agnews(opt["data"], mode="test", indices=range(10))
    experiment_type = "model"

    if experiment_type == "batch":

        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=analyzer.args.batch_size, shuffle=False)

        test_grad_dict = {}
        # here we should extract all the intermediate features and individual gradients of training data
        for _, (idx, data, label) in enumerate(test_loader):
            torch.cuda.empty_cache()
            _test_grad = _get_data_artifacts(analyzer, data, label, method="naive")[0]

            for layer_name in _test_grad:
                if analyzer.args.dataset in ["cifar10", "imagenet"]:
                    _test_grad[layer_name] = _test_grad[layer_name].mean(0)
                test_grad_dict[layer_name] = test_grad_dict.get(layer_name, 0) + _test_grad[layer_name] * len(idx)

        for layer_name in test_grad_dict:
            test_grad_dict[layer_name] /= len(test_dataset)

        test_artifacts_list = [test_grad_dict]
        total_meta_gradient = query_batch(analyzer, test_artifacts_list, opt.method)
    else:
        total_meta_gradient = 0

        for i in range(len(test_dataset)):
            print (i)
            idx, data, label = test_dataset[i]

            if opt["data"]["data_name"] == "AGNews":
                data = {key: value.unsqueeze(0) for key, value in data.items()}
                label = label.unsqueeze(0)
            else:
                data = data.unsqueeze(0)
                label = torch.tensor(label)

            test_artifacts_list = _get_data_artifacts(analyzer, data, label, opt["system"]["method"])

            meta_gradient = query(analyzer, test_artifacts_list, method=opt["system"]["method"])
            total_meta_gradient += meta_gradient

    print (total_meta_gradient)

    logger.print()

    print (opt)

    # file_name = opt.output_path +"/hparams.pickle"
    # with open(file_name, "wb") as handle:
    #     pickle.dump(vars(opt), handle)
    #
    # print (total_meta_gradient)
    # file_name = opt.output_path + "/total_meta_gradient_{}.pickle".format(opt.method)
    # with open(file_name, 'wb') as handle:
    #     pickle.dump(total_meta_gradient, handle)

    # print (e2e_query_timer)
    # file_name = opt.output_path + "/e2e_query_timer_{}.pickle".format(opt.method)
    # with open(file_name, 'wb') as handle:
    #     pickle.dump(e2e_query_timer, handle)
    #
    # print (cal_query_timer_per_layer)
    # file_name = opt.output_path + "/calculation_query_timer_{}.pickle".format(opt.method)
    # with open(file_name, 'wb') as handle:
    #     pickle.dump(cal_query_timer_per_layer, handle)
    #
    # print (io_timer_per_layer)
    # file_name = opt.output_path + "/io_query_timer_{}.pickle".format(opt.method)
    # with open(file_name, 'wb') as handle:
    #     pickle.dump(io_timer_per_layer, handle)





if __name__ == "__main__":
    # main()
    config = test_merge_config()
    main(config)