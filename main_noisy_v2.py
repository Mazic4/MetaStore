import argparse
import os.path
import random
import pickle
import sys
import timeit

import numpy as np

sys.path.append('../')
print(sys.path)

import torch
import torch.nn as nn
import torch.optim as optim

from model_trainers.models import VGG16, Customized_Bert_Model
from utils.data_loader import *
from utils.utils import *
from utils.grad_to_feats import *
from _analyzer import Analyzer
from artifacts_storage.artifacts_store import store_traindata_artifacts
from query.query_batch_executor import query_batch


def main(noisy_data_limit, method):
    # TODO: remove this, and create a test script.

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1, help="sgd: learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="sgd: momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="sgd: weight_decay")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")

    parser.add_argument("--output_path", type=str, default="./experiment_noisy_result_debug",
                        help="the path to store results")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")

    parser.add_argument("--analyzed_epoch", type=int, default=2, help="the epoch number of the analyzed model ckp.")
    parser.add_argument("--extract_artifacts", type=bool, default=True, help="extract and store artifacts.")
    parser.add_argument("--device", type=str, default="cuda", help="use cpu or gpu to calculate meta-gradient.")

    parser.add_argument("--experiment_type", type=str, default="batch", choices=["model", "layer", "batch", "ablation"],
                        help="experiment type")
    parser.add_argument("--method", type=str, default="half", choices=["naive", "ted", "half", "reproduce", "recon"])
    parser.add_argument("--precision_mode", type=str, default="normal", choices=["normal", "quant"])
    parser.add_argument("--float_precision", type=int, default=1, choices=[1, 2, 4, 6, 8])
    parser.add_argument("--hidden_size", type=int, default=768, help="number of output features in fcn for cifar")
    parser.add_argument("--analyze_layer_type", type=str, default="linear", help="which type of layers to analyze")

    parser.add_argument("--max_store_batch_size", type=int, default=100, help="batch_size")

    parser.add_argument("--gpu_memory_threshold", type=float, default=0.2, help="the ratio of gpu memory used to cache")
    parser.add_argument("--num_samples", type=int, default=50000, help="the number of traning samples analyzed")
    parser.add_argument("--num_query", type=int, default=100, help="the number of queries performed")
    # parser.add_argument("--gpu_cache", type=bool, default=False, help="if use gpu to cache artifacts")
    parser.add_argument("--num_pseudo_samples", type=int, default=10,
                        help="number of pseudo samples if use query batch")
    parser.add_argument("--noisy_data_limit", type=int, default=0)


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

    opt = parser.parse_args()

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    print("Set seed :", seed)

    experiment_result_log_base = "experiment_type_{}_method_{}_dataset_{}_num_samples_{}_num_query_{}".format(
                opt.experiment_type,
                opt.method,
                opt.dataset,
                opt.num_samples,
                opt.num_query
            )

    opt.output_path = os.path.join(opt.output_path, experiment_result_log_base)

    opt.noisy_data_limit = noisy_data_limit

    analyzer = Analyzer(opt)

    model = VGG16()
    model.load_state_dict(torch.load('./model_trainers/models/vgg16_models/cifar10_state_dict_noisy_v2_{}_{}.pth'.format(noisy_data_limit, opt.analyzed_epoch)))

    model = model.cuda()

    target_classes = [1,2,3,4,5,6,7,8,9]
    noisy_class = [(0, 1)]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([32, 32]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize([32, 32]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # create noisy labels
    # create noisy labels
    train_labels, test_labels = [], []
    train_data_indices, test_data_indices = [], []
    noisy_data_counter = 0
    org_train_dataset = torchvision.datasets.CIFAR10('data', transform=transform, train=True, download=True)
    for i in range(len(org_train_dataset)):
        org_label = org_train_dataset[i][1]
        if org_label in target_classes:
            train_labels.append(org_label)
            train_data_indices.append(i)
        elif org_label == noisy_class[0][0]:
            if noisy_data_counter < noisy_data_limit:
                train_labels.append(noisy_class[0][1])
                noisy_data_counter += 1
            else:
                train_labels.append(org_label)
            train_data_indices.append(i)

    org_test_dataset = torchvision.datasets.CIFAR10('data', transform=transform, train=False, download=True)
    for i in range(len(org_test_dataset))[:1000]:
        org_label = org_test_dataset[i][1]
        test_labels.append(org_label)
        test_data_indices.append(i)

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    analyzer.train_dataset = Customized_Dataset_noisy(org_train_dataset, train_labels, index=train_data_indices)
    test_dataset = Customized_Dataset_noisy(org_test_dataset, test_labels, index=test_data_indices)

    analyzer.train_loader = torch.utils.data.DataLoader(analyzer.train_dataset,
                                                    batch_size=opt.batch_size, shuffle=True)

    analyzer.num_batches = len(analyzer.train_dataset) // analyzer.args.max_store_batch_size

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

    loss_func = nn.CrossEntropyLoss()

    analyzer.load_models(model, optimizer)
    analyzer.set_loss_func(loss_func)

    # store_traindata_artifacts(analyzer, method="ted")
    # store_traindata_artifacts(analyzer, method="naive")

    analyzer.get_gpu_size()
    analyzer.get_artifacts_size()

    total_meta_gradient = 0



    noisy_labels = np.load(
        "./model_trainers/models/vgg16_models/cifar_noisy_labels_{}_v2.npy".format(noisy_data_limit))

    noisy_sample_indices = []
    org_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True)
    for i in range(len(org_dataset)):
        if org_dataset[i][1] == noisy_class[0][0] and len(noisy_sample_indices) < noisy_data_limit:
            noisy_sample_indices.append(i)

    noisy_sample_indices = np.array(noisy_sample_indices)

    num_noisy_labels = len(noisy_sample_indices)
    print(num_noisy_labels)

    if method == "meta_grad":

        start_time = timeit.default_timer()

        model = VGG16()
        model.load_state_dict(torch.load(
            './model_trainers/models/vgg16_models/cifar10_state_dict_noisy_v2_{}_{}.pth'.format(noisy_data_limit,
                                                                                                2)))

        model = model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=opt.lr)
        loss_func = nn.CrossEntropyLoss()
        analyzer.load_models(model, optimizer)

        _total_meta_gradient, e2e_query_timer, cal_query_timer_per_layer, io_timer_per_layer = query_batch(analyzer,
                                                                                                          test_dataset,
                                                                                                          method = "half")

        print (_total_meta_gradient)

        total_meta_gradient += _total_meta_gradient

        value = total_meta_gradient.cpu()

        print("end to end time: ", timeit.default_timer() - start_time)

    elif method == "grad_shap":

        start_time = timeit.default_timer()

        round = 0
        converge_flag = False

        grad_shap = 0

        for epoch in range(5):
            torch.cuda.empty_cache()
            model = VGG16()
            model.load_state_dict(torch.load(
                './model_trainers/models/vgg16_models/cifar10_state_dict_noisy_v2_{}_{}.pth'.format(noisy_data_limit,
                                                                                                    epoch)))

            model = model.cuda()
            optimizer = optim.SGD(model.parameters(), lr=opt.lr)
            loss_func = nn.CrossEntropyLoss()
            analyzer.load_models(model, optimizer)

            round += 1
            print("round", round)
            _shap_total_meta_gradient, e2e_query_timer, cal_query_timer_per_layer, io_timer_per_layer = query_batch(
                analyzer,
                test_dataset,
                method="naive")

            new_grad_shap = 1 / round * _shap_total_meta_gradient + (round - 1) / round * grad_shap
            converge_flag = coverge_func(new_grad_shap, grad_shap)
            grad_shap = new_grad_shap[:]
            if converge_flag:
                break

            truth_top_k = noisy_sample_indices
            for i in [noisy_data_limit]:
                print(i)
                if method == "small_loss":
                    pred_top_k = torch.argsort(grad_shap[train_data_indices])[-i:]
                else:
                    pred_top_k = torch.argsort(grad_shap[train_data_indices])[:i]

                print(len(np.intersect1d(truth_top_k, pred_top_k.cpu())))

        value = grad_shap.cpu()

        print("end to end time: ", timeit.default_timer() - start_time)

    elif method == "small_loss":

        model = VGG16()
        model.load_state_dict(torch.load(
            './model_trainers/models/vgg16_models/cifar10_state_dict_noisy_v2_{}_{}.pth'.format(noisy_data_limit,
                                                                                                4)))

        model = model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=opt.lr)
        loss_func = nn.CrossEntropyLoss()
        analyzer.load_models(model, optimizer)

        start_time = timeit.default_timer()
        s_loss_trick = small_loss_trick(analyzer, noisy_labels)
        print ("end to end time: ",  timeit.default_timer() - start_time)

        print(s_loss_trick)

        value = s_loss_trick.cpu()

    truth_top_k = noisy_sample_indices
    for i in [noisy_data_limit]:
        print (i)
        if method == "small_loss":
            pred_top_k = torch.argsort(value[train_data_indices])[-i:]
        else:
            pred_top_k = torch.argsort(value[train_data_indices])[:i]

        print (len(np.intersect1d(truth_top_k, pred_top_k)))

    # data_feats = test(analyzer)
    #
    # lof_res = detect_noisy_sample_baselines(data_feats)
    # print (lof_res)

def small_loss_trick(self, noisy_labels):
    train_loss = torch.zeros(50000)
    loss_func = nn.CrossEntropyLoss(reduction='none')
    for batch_idx, (idx, data, label) in enumerate(self.train_loader):
        logits = self.model(data.cuda())
        loss = loss_func(logits, label.cuda())
        train_loss[idx] = loss.cpu().detach()

    return train_loss

def test(self):
    data_feats = torch.zeros(50000, 512)

    for batch_idx, (idx, data, label) in enumerate(self.train_loader):
        feats = self.model(data.cuda(), feature = True).reshape(len(idx),512)
        data_feats[idx] = feats.cpu().detach()


    return data_feats

def coverge_func(new_grad_shap, grad_shap, threshold = 0.001):
    # print (new_grad_shap, grad_shap)
    diff = (new_grad_shap-grad_shap)**2
    flag = torch.sum(diff)**0.5 < (threshold)
    return flag


if __name__ == "__main__":
    # import shutil
    # shutil.rmtree("./artifacts/cifar10_artifacts_layer")
    for noisy_data_limit in [1000]:
        # for method in ["meta_grad", "grad_shap", "small_loss"]:
        for method in ["grad_shap"]:
            print (noisy_data_limit, method)
            main(noisy_data_limit, method)