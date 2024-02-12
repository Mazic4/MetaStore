import os
import sys
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision

from backpack import backpack, extend
from backpack.extensions import BatchGrad, BatchL2Grad
from backpack.extensions.firstorder.batch_grad import *

from models import Network, VGG16
from ..utils.data_loader import *
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self, args):
        self.args = args

        self.build_model()
        self.get_data()

    def build_model(self):

        self.model = VGG16().cuda()
        self.model = torch.nn.DataParallel(self.model)

        self.optimizer = optim.SGD(self.model.parameters(
        ), lr=self.args.lr, momentum=self.args.momentum)
        self.loss_func = nn.CrossEntropyLoss()

    def get_data(self, dataset="cifar10"):

        if dataset == "cifar10":
            self.transform = torchvision.transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                torchvision.transforms.Resize([32, 32]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            self.transform_test = torchvision.transforms.Compose([
                torchvision.transforms.Resize([32, 32]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            # self.train_dataset = datasets.MNIST('data', transform= self.transform, train=True, download=True)
            self.train_dataset = Customized_Dataset(
                torchvision.datasets.CIFAR10('data', transform=self.transform, train=True, download=True))
            self.test_dataset = Customized_Dataset(
                torchvision.datasets.CIFAR10('data', transform=self.transform_test, train=False, download=True))

            self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.args.batch_size, shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                           batch_size=self.args.batch_size, shuffle=True)

        else:
            print("Using MNIST as dataset")
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])

            self.train_dataset = Customized_Dataset(
                torchvision.datasets.MNIST('/dataset/', train=True, download=True, transform=transform))

            self.test_dataset = Customized_Dataset(
                torchvision.datasets.MNIST('/dataset/', train=False, download=True, transform=transform))

            self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.args.batch_size, shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                           batch_size=self.args.batch_size, shuffle=True)

    def train(self):

        self.model.train()

        train_loss = 0

        for _, (idx, data, label) in tqdm(enumerate(self.train_loader)):

            data, label = Variable(data).cuda(), Variable(label).cuda()

            output_logits = self.model(data)
            loss = self.loss_func(output_logits, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss/len(self.train_loader)

        return train_loss

    def eval(self):

        self.model.eval()

        test_acc = 0
        test_loss = 0

        for _, (idx, data, label) in enumerate(self.test_loader):

            data, label = Variable(data).cuda(), Variable(label).cuda()

            output_logits = self.model(data)
            loss = self.loss_func(output_logits, label)
            test_acc += torch.sum(torch.argmax(output_logits, 1)
                                  == label.reshape(-1,))
            test_loss += loss.item() * len(self.test_loader)

        return test_acc/len(self.test_dataset), test_loss/len(self.test_dataset)

    def main(self, mode="train"):

        for e in range(self.args.n_epochs):
            train_loss = self.train()
            test_acc, test_loss = self.eval()
            print('Epoch: {} - Train Loss {:.6f}, Test Loss {:.6f}, Test Acc: {:.6f}'.format(
                e, train_loss, test_loss, test_acc))

            torch.save(self.model.module.state_dict(),
                       './models/vgg16_models/cifar10_state_dict_finetune_{}.pth'.format(e))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", type=int, default=20,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int,
                        default=100, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="sgd: learning rate")
    parser.add_argument("--momentum", type=float,
                        default=0.9, help="sgd: momentum")
    parser.add_argument("--weight_decay", type=float,
                        default=5e-4, help="sgd: weight_decay")
    parser.add_argument("--n_cpu", type=int, default=1,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="number of output features in fcn for cifar")

    opt = parser.parse_args()

    print(opt)

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print("Set seed :", seed)

    trainer = Trainer(opt)
    trainer.main()
