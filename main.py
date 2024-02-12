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
from utils.data_loader import *
from utils.utils import *
from utils.grad_to_feats import *
from _analyzer import Analyzer
from artifacts_calculator.artifacts_collector import _get_data_artifacts
from artifacts_storage.artifacts_store import store_traindata_artifacts
from query.query_executor import query
from query.query_batch_executor import query_batch

from runtime_log import logger


def main():
    # TODO: remove this, and create a test script.

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1, help="sgd: learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="sgd: momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="sgd: weight_decay")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")

    parser.add_argument("--output_path", type=str, default="./experiment_result_test1",
                        help="the path to store results")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")

    parser.add_argument("--analyzed_epoch", type=int, default=0, help="the epoch number of the analyzed model ckp.")
    parser.add_argument("--extract_artifacts", type=bool, default=True, help="extract and store artifacts.")
    parser.add_argument("--device", type=str, default="cuda", help="use cpu or gpu to calculate meta-gradient.")

    parser.add_argument("--experiment_type", type=str, default="model", choices=["model", "layer", "batch", "ablation"],
                        help="experiment type")
    parser.add_argument("--method", type=str, default="ted", choices=["naive", "ted", "half", "reproduce", "recon", "iter"])
    parser.add_argument("--precision_mode", type=str, default="normal", choices=["normal", "quant"])
    parser.add_argument("--float_precision", type=str, default=1, choices=["torch.quint8", "torch.qint8", "torch.qint32", "torch.float16"])
    parser.add_argument("--hidden_size", type=int, default=None, help="number of output features in fcn for cifar")
    parser.add_argument("--analyze_layer_type", type=str, default="linear", help="which type of layers to analyze")

    parser.add_argument("--collect_artifacts", type=bool, default=True, help="collect artifacts or not")
    parser.add_argument("--max_store_batch_size", type=int, default=100, help="batch_size")

    parser.add_argument("--gpu_memory_threshold", type=float, default=0.05, help="the ratio of gpu memory used to cache")
    parser.add_argument("--num_samples", type=int, default=500, help="the number of traning samples analyzed")
    parser.add_argument("--num_query", type=int, default=10, help="the number of queries performed")
    parser.add_argument("--num_pseudo_samples", type=int, default=10,
                        help="number of pseudo samples if use query batch")

    # This is used for the AGNews dataset.
    parser.add_argument("--train_file", type=str,
                        default="./data/AGNews/train_data.json",
                        help="training set")
    parser.add_argument("--dev_file", type=str,
                        default="./data/AGNews/dev_data.json", help="dev set")
    parser.add_argument("--test_file", type=str,
                        default="./data/AGNews/test_data.json", help="test set")
    parser.add_argument("--unlabeled_file", type=str,
                        default="./data/AGNews/unlabeled_data.json", help="unlabeled set")
    parser.add_argument("--bert_hidden_size", type=int, default=768, help="number of output features in fcn for cifar")

    opt = parser.parse_args()

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    print("Set seed :", seed)

    sample_index = np.arange(opt.num_samples)

    _, _, train_dataset, test_dataset = get_dataloaders(opt, opt.dataset, train_indices=sample_index,
                                                        test_indices=np.arange(opt.num_query))

    experiment_result_log_base = "experiment_type_{}_method_{}_dataset_{}_num_samples_{}_num_query_{}".format(
                opt.experiment_type,
                opt.method,
                opt.dataset,
                opt.num_samples,
                opt.num_query
            )

    if not os.path.exists(opt.output_path):
        os.mkdir(opt.output_path)

    opt.output_path = os.path.join(opt.output_path, experiment_result_log_base)
    if opt.precision_mode == "quant":
        opt.output_path += "_quant"
        opt.output_path += "_{}".format(opt.float_precision)

    if not os.path.exists(opt.output_path):
        os.mkdir(opt.output_path)

    analyzer = Analyzer(opt)

    analyzer.set_traindata(train_dataset)

    if opt.dataset == "cifar10":
        model = VGG16()
        model.load_state_dict(
            torch.load('../ted/models/vgg16_models/cifar10_state_dict_finetune_{}.pth'.format(opt.analyzed_epoch)))

    elif opt.dataset == "AGNews":
        model = Customized_Bert_Model(num_classes=4).cuda()
        model.load_state_dict(
            torch.load('/home/zhanghuayi01/TED/ted/models/bert_models/agnews_state_dict_finetune_{}_{}.pth'.
                       format(opt.analyzed_epoch, opt.bert_hidden_size)), strict=False)
    else:
        raise NotImplementedError("not implemented dataset")

    if opt.device == "cuda":
        model = model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    loss_func = nn.CrossEntropyLoss()
    analyzer.load_models(model, optimizer)
    analyzer.set_loss_func(loss_func)

    if analyzer.args.collect_artifacts:
        store_traindata_artifacts(analyzer, method=opt.method)

    analyzer.get_gpu_size()
    analyzer.get_artifacts_size()

    if opt.experiment_type == "batch":

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

            if opt.dataset == "AGNews":
                data = {key: value.unsqueeze(0) for key, value in data.items()}
                label = label.unsqueeze(0)
            else:
                data = data.unsqueeze(0)
                label = torch.tensor(label)

            test_artifacts_list = _get_data_artifacts(analyzer, data, label, opt.method)

            meta_gradient = query(analyzer, test_artifacts_list, method=opt.method)
            total_meta_gradient += meta_gradient

    print (total_meta_gradient)

    print (logger)

    print (vars(opt))
    file_name = opt.output_path +"/hparams.pickle"
    with open(file_name, "wb") as handle:
        pickle.dump(vars(opt), handle)

    print (total_meta_gradient)
    file_name = opt.output_path + "/total_meta_gradient_{}.pickle".format(opt.method)
    with open(file_name, 'wb') as handle:
        pickle.dump(total_meta_gradient, handle)

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
    main()