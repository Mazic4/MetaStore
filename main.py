import argparse
import random
import sys

sys.path.append('../')
print(sys.path)

from utils.data_loader_new import *
from utils.grad_to_feats import *
from analyzer import Analyzer
from artifacts_calculator.artifacts_collector import _get_data_artifacts
from query.query_executor import query
from query.query_batch_executor import query_batch

from config import merge_configs
from runtime_log import logger

def main(opt):

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    print("Set seed :", seed)

    analyzer = Analyzer(opt)

    if opt["data"]["data_name"] == "cifar10":
        test_loader, test_dataset = get_dataloader_cifar(opt["data"], mode="test", indices=range(opt["system"]["num_query"]))
    elif opt["data"]["data_name"] == "imagenet":
        test_loader, test_dataset = get_dataloader_imagenet(opt["data"], mode="test", indices=range(opt["system"]["num_query"]))
    elif opt["data"]["data_name"] == "AGNews":
        test_loader, test_dataset = get_dataloader_agnews(opt["data"], mode="test", indices=range(opt["system"]["num_query"]))
    else:
        raise NotImplementedError

    if analyzer.method in ["naive", "ted"]:
        operator_type = "p2p"
    else:
        operator_type = "p2b"

    if operator_type == "p2b":

        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=analyzer.args["data"]["batch_size"], shuffle=False)

        test_grad_dict = {}
        # here we should extract all the intermediate features and individual gradients of training data
        for _, (idx, data, label) in enumerate(test_loader):
            torch.cuda.empty_cache()
            _test_grad = _get_data_artifacts(analyzer, data, label, method="naive")[0]

            for layer_name in _test_grad:
                if analyzer.args["data"]["data_name"] in ["cifar10", "imagenet"]:
                    _test_grad[layer_name] = _test_grad[layer_name].mean(0)
                test_grad_dict[layer_name] = test_grad_dict.get(layer_name, 0) + _test_grad[layer_name] * len(idx)

        for layer_name in test_grad_dict:
            test_grad_dict[layer_name] /= len(test_dataset)

        test_artifacts_list = [test_grad_dict]
        total_meta_gradient = query_batch(analyzer, test_artifacts_list, opt["system"]["method"])
    else:
        total_meta_gradient = 0

        for i in range(len(test_dataset)):
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

    logger.save(_log_base_dir="./log", config=opt, meta_gradient=total_meta_gradient)
    logger.print()


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument("--config", type=str, default="./config.yaml", help="config_path")
        parser.add_argument("--method", type=str, default="ted", help="method")
        parser.add_argument("--target_model", type=str, default="ResNet50", help="model")
        parser.add_argument("--data", type=str, default="imagenet", help="data")
        parser.add_argument("--num_analyzed_samples", type=int, default=100, help="num of analyzed data")
        parser.add_argument("--num_query", type=int, default=1, help="num of analyzed data")

        return parser.parse_args()

    args = parse_args()
    config = merge_configs(args)
    main(config)