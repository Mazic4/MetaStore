import argparse
import os.path
import random
import pickle
import sys

import numpy as np

sys.path.append('../')
print(sys.path)

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer,AutoModel,BertForSequenceClassification

from model_trainers.models import VGG16, Customized_Bert_Model
from utils.data_loader import *
from utils.utils import *
from utils.grad_to_feats import *
from _analyzer import Analyzer
from artifacts_storage.artifacts_store import store_traindata_artifacts
from artifacts_loader.artifacts_loader import load_data_artifacts
from query.query_executor import query
from query.query_batch_executor import query_batch


def main():
    # TODO: remove this, and create a test script.

    parser = argparse.ArgumentParser()

    # parser.add_argument("--artifacts_log_path", type=str, default="./artifacts/cifar10_conv_artifacts_org")

    parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1, help="sgd: learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="sgd: momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="sgd: weight_decay")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")

    parser.add_argument("--output_path", type=str, default="./experiment_noisy_result_debug",
                        help="the path to store results")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")

    parser.add_argument("--analyzed_epoch", type=int, default=0, help="the epoch number of the analyzed model ckp.")
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

    # sample_index = np.random.choice(np.arange(50000), size=10000, replace=False)
    sample_index = np.arange(opt.num_samples)

    _, _, train_dataset, test_dataset = get_dataloaders(opt, opt.dataset, train_indices=sample_index,
                                                        test_indices=np.arange(opt.num_query),
                                                        noisy_ratio= 0.05)

    experiment_result_log_base = "experiment_type_{}_method_{}_dataset_{}_num_samples_{}_num_query_{}".format(
                opt.experiment_type,
                opt.method,
                opt.dataset,
                opt.num_samples,
                opt.num_query
            )

    opt.output_path = os.path.join(opt.output_path, experiment_result_log_base)

    analyzer = Analyzer(opt)

    analyzer.set_traindata(train_dataset)

    if opt.dataset == "cifar10":
        model = VGG16()
        model.load_state_dict(torch.load('./model_trainers/models/vgg16_models/cifar10_state_dict_noisy_0.05_{}.pth'.format(opt.analyzed_epoch)))

    model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

    loss_func = nn.CrossEntropyLoss()

    analyzer.load_models(model, optimizer)
    analyzer.set_loss_func(loss_func)

    store_traindata_artifacts(analyzer, method="ted")
    # store_traindata_artifacts(analyzer, method="naive")

    analyzer.get_gpu_size()
    analyzer.get_artifacts_size()

    if opt.dataset == "cifar10":

        total_meta_gradient = 0

        for epoch in range(0, 1):

            model = VGG16()
            model.load_state_dict(torch.load(
                './model_trainers/models/vgg16_models/cifar10_state_dict_noisy_0.05_{}.pth'.format(
                    epoch)))

            model = model.cuda()
            optimizer = optim.SGD(model.parameters(), lr=opt.lr)
            loss_func = nn.CrossEntropyLoss()
            analyzer.load_models(model, optimizer)

            _total_meta_gradient, e2e_query_timer, cal_query_timer_per_layer, io_timer_per_layer = query_batch(analyzer,
                                                                                                              test_dataset,
                                                                                                              opt.method)

            total_meta_gradient += _total_meta_gradient

        np.save("first_layer_meta_gradient.npy", total_meta_gradient.cpu().numpy())

        first_layer_meta_gradient = torch.tensor(np.load("first_layer_meta_gradient.npy")).cuda()
        # total_meta_gradient += first_layer_meta_gradient

        noisy_labels = np.load("./model_trainers/models/vgg16_models/cifar_noisy_labels_0.05.npy")

        labels = []
        org_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True)
        for i in range(len(org_dataset)):
            labels.append(org_dataset[i][1])

        labels = np.array(labels)

        _binary_noisy_labels = noisy_labels != labels

        # from sklearn.metrics import f1_score
        #
        num_noisy_labels = np.sum(_binary_noisy_labels)
        print (num_noisy_labels)
        # pred_top_k = np.argsort(total_meta_gradient.cpu().numpy())[:500]
        # y_pred = np.zeros(len(_binary_noisy_labels))
        # y_pred[pred_top_k] = 1
        #
        # print (_binary_noisy_labels, y_pred)
        # print (f1_score(_binary_noisy_labels, y_pred))

        s_loss_trick = small_loss_trick(analyzer, noisy_labels)

        print (s_loss_trick)

        truth_top_k = np.argsort(_binary_noisy_labels)[-num_noisy_labels:]
        for i in [100, 500, 1000, 2000, 2500, 5000, 10000, 20000, 50000]:
            print (i)
            pred_top_k = np.argsort(s_loss_trick)[-i:]

            print (len(np.intersect1d(truth_top_k, pred_top_k)))



        for i in [100, 500, 1000, 2000, 2500, 5000, 10000, 20000, 50000]:
            print (i)
            pred_top_k = np.argsort(total_meta_gradient.cpu().numpy())[:i]
            print (len(np.intersect1d(truth_top_k, pred_top_k)))

            pred_top_k = np.argsort(total_meta_gradient.cpu().numpy())[-i:]
            print(len(np.intersect1d(truth_top_k, pred_top_k)))

    elif opt.dataset == "AGNews":

        # true_labels = []
        #
        # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # org_lab_dataset, _, _, _ = get_dataset_text(opt, tokenizer)
        #
        # for i in range(len(org_lab_dataset)):
        #     true_labels.append(org_lab_dataset[i][2].item())
        # true_labels = np.array(true_labels)

        # np.save("./model_trainers/models/bert_models/agnews_true_labels.npy", true_labels)
        noisy_labels = np.load("./model_trainers/models/bert_models/agnews_noisy_labels_0.2.npy")
        true_labels = np.load("./model_trainers/models/bert_models/agnews_true_labels.npy")

        _binary_noisy_labels = noisy_labels == true_labels
        num_noisy_labels = len(true_labels) - np.sum(_binary_noisy_labels)

        print (noisy_labels, true_labels)

        print (num_noisy_labels)
        print (len(true_labels))

        for i in [100, 500, 1000, 2000, 5000, 10000]:
            print (i)
            truth_top_k = np.argsort(_binary_noisy_labels)[:num_noisy_labels]
            pred_top_k = np.argsort(total_meta_gradient.cpu().numpy())[:i]

            print (len(np.intersect1d(truth_top_k, pred_top_k)))

    # data_feats = test(analyzer)
    #
    # lof_res = detect_noisy_sample_baselines(data_feats)
    # print (lof_res)

def small_loss_trick(self, noisy_labels):
    train_loss = torch.zeros(len(self.train_dataset))
    loss_func = nn.CrossEntropyLoss(reduction='none')
    for batch_idx, (idx, data, label) in enumerate(self.train_loader):
        logits = self.model(data.cuda())
        loss = loss_func(logits, label.cuda())
        train_loss[idx] = loss.cpu().detach()

    return train_loss

def test(self):
    data_feats = torch.zeros(len(self.train_dataset), 512)

    for batch_idx, (idx, data, label) in enumerate(self.train_loader):
        feats = self.model(data.cuda(), feature = True).reshape(len(idx),512)
        data_feats[idx] = feats.cpu().detach()


    return data_feats


if __name__ == "__main__":
    # import shutil
    # shutil.rmtree("./artifacts/cifar10_artifacts_layer")
    main()