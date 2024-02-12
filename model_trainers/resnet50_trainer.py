import os
import sys
import argparse
import random
import subprocess
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision

from backpack import backpack, extend
from backpack.extensions import BatchGrad, BatchL2Grad
from backpack.extensions.firstorder.batch_grad import *

from models import Network, VGG16, ResNet50, Identity
from ted_v2.utils.data_loader import *
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self, args):
        self.args = args

        self.build_model()
        self.get_data()

    def build_model(self):

        self.model = ResNet50().cuda()
        if self.args.hidden_size != None:
            self.model.append(layer_type="linear", hidden_size=self.args.hidden_size)
        self.model = torch.nn.DataParallel(self.model)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.loss_func = nn.CrossEntropyLoss()

    def get_data(self):

        data_dir = './data/'
        subprocess.Popen('pwd')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        if not os.path.exists(data_dir + 'imagenet64.tar'):
            print("Downloading archive...")
            os.chdir(data_dir)

            os.system("wget https://pjreddie.com/media/files/imagenet64.tar")

            os.chdir(data_dir)
            print("Uncompressing...")
            os.system("tar -xf imagenet64.tar")
            os.chdir("/home/zhanghuayi01/TED/ted_v2")
        else:
            os.chdir("/home/zhanghuayi01/TED/ted_v2")

        print("Data ready!")

        # Data augmentation transformations. Not for Testing!
        transform_train = transforms.Compose([
            transforms.Resize(64),  # Takes images smaller than 64 and enlarges them
            transforms.RandomCrop(64, padding=4, padding_mode='edge'),  # Take 64x64 crops from 72x72 padded images
            transforms.RandomHorizontalFlip(),  # 50% of time flip image along y-axis
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.train_dataset = Customized_Dataset(torchvision.datasets.ImageFolder(root=data_dir + 'imagenet64/train/', transform=transform_train))
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=128, shuffle=True, num_workers=2)

        self.test_dataset = Customized_Dataset(torchvision.datasets.ImageFolder(root=data_dir + 'imagenet64/val/', transform=transform_test))
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=2)

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

            train_loss += loss / len(self.train_loader)

            if _%100 == 0:
                print ("------Training Info------")
                print ("training step: ", _)
                print ("training loss: ", loss)

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
                                  == label.reshape(-1, ))
            test_loss += loss.item() * len(self.test_loader)

        return test_acc / len(self.test_dataset), test_loss / len(self.test_dataset)

    def main(self, mode="train"):

        save_model_dir = "./models/resnet50_models"

        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)

        for e in range(self.args.n_epochs):
            train_loss = self.train()
            test_acc, test_loss = self.eval()
            print('Epoch: {} - Train Loss {:.6f}, Test Loss {:.6f}, Test Acc: {:.6f}'.format(
                e, train_loss, test_loss, test_acc))

            if self.args.hidden_size is None:
                torch.save(self.model.module.state_dict(),
                       './models/resnet50_models/imagenet_state_dict_finetune_{}.pth'.format(e))
            else:
                torch.save(self.model.module.state_dict(),
                           './models/resnet50_state_dict_finetune_{}_{}.pth'.format(e, opt.hidden_size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", type=int, default=1,
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