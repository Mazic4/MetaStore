import sys
import random
import argparse
from tqdm import tqdm
from collections import defaultdict
import pickle
import copy
from tqdm import tqdm, trange

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd import Variable

from transformers import AdamW, AutoTokenizer, BertConfig, get_linear_schedule_with_warmup, AutoModel, BertModel
from transformers import BertForSequenceClassification

from ted_v3.utils.data_loader import get_dataset, get_dataset_text, get_dataset_features

from utils import *
from models import *

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class Trainer_text(object):

    def __init__(self, args):
        self.args = args

        self.build_data()
        self.build_model()
        self.lab_loss = 0
        self.lab_acc = 0

        self.max_scale = self.args.max_scale * len(self.lab_dataset) / len(self.unlabeled)

    def build_model(self):

        self.device = "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"
        # self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.num_class).cuda()

        self.model = Customized_Bert_Model(hidden_size = self.args.hidden_size, num_classes = 4).cuda()

        # # Accessing the model configuration
        # configuration = model.config

        # self.model = nn.DataParallel(self.model)

        no_decay = ['bias', 'LayerNorm.weight']

        # for param in self.model.encoder.parameters():
        #     param.requires_grad = False

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        # self.optimizer = AdamW(self.model.params(), lr=self.args.lr)
        self.naive_lab_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.lab_loss_f = torch.nn.CrossEntropyLoss()

    def build_data(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.lab_dataset, self.unlabeled, self.test_dataset, self.dev_dataset = get_dataset_text(self.args, tokenizer)

        self.train_dataloader = DataLoader(self.lab_dataset, shuffle=True, batch_size=self.args.batch_size)
        self.unl_data_loader = DataLoader(self.unlabeled, shuffle=True, batch_size=self.args.unl_batch_size)
        self.test_loader = DataLoader(self.test_dataset, shuffle=True, batch_size=self.args.batch_size)

        n = torch.tensor(list(zip(*self.lab_dataset))[2])
        self.num_class = len(torch.unique(n))

    def train(self, epoch):

        self.model.train()

        # Prepare optimizer and schedule (linear warmup and decay)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,num_training_steps=max_iters)
        # Train!

        self.lab_loss = 0
        self.lab_acc = 0

        for _, (_idx, lab_inputs, lab_true_label) in tqdm(enumerate(self.train_dataloader)):

            lab_inputs = {key: to_var(value, False) for key, value in lab_inputs.items()}
            lab_true_label = lab_true_label.cuda()

            # train
            output = self.model(**lab_inputs)
            lab_logits = self.model(**lab_inputs).logits

            loss = self.naive_lab_loss(lab_logits, lab_true_label).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def eval(self):

        self.model.eval()

        test_acc = 0
        test_loss = 0

        test_acc_l = []
        for _, (idx, inputs, true_labels) in enumerate(self.test_loader):

            inputs = {key: to_var(value, False) for key, value in inputs.items()}
            true_labels = true_labels.cuda()

            # train
            output_logits = self.model(**inputs).logits
            loss = self.lab_loss_f(output_logits, true_labels)
            test_acc += torch.sum(torch.argmax(output_logits, dim=1) == true_labels).item()
            test_loss += loss.item()
            print(test_loss)

        return test_acc / len(self.test_dataset), test_loss / len(self.test_dataset)

    def main(self):

        log = {"unl_acc": [], 'unl_loss': [], "entrs": [], "lab_acc": [], "lab_loss": [], "test_acc": [],
               "test_loss": [], "meta_gradient_inter": [], "meta_gradient_intra": []}

        for epoch in range(self.args.n_epochs):
            print('epoch:', epoch)
            self.train(epoch)

            print('eval on test')
            t_acc, t_loss = self.eval()
            log['test_acc'].append(t_acc)
            log['test_loss'].append(t_loss)

            print (t_acc, t_loss)

            pickle.dump(log, open("agnews_nometa_clean{}_{}_log.pkl".format(self.args.clean, self.args.unl_batch_size),
                                  "wb"))

            torch.save(self.model.state_dict(),
                       '/home/zhanghuayi01/TED/ted/models/bert_models/agnews_state_dict_finetune_{}_{}.pth'
                       .format(epoch, self.model.bert.config.hidden_size))

        print('finished traing and saved log into .pkl file')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_text", type=bool, default=True, help="experiment with text or image")
    parser.add_argument("--log_path", type=str, default="./experiment_results/")
    parser.add_argument("--sample_interval", type=int, default=1, help="interval betwen log info")

    parser.add_argument("--cal_meta", type=bool, default=False, help="Calculate meta-gradient or not.")
    parser.add_argument("--file_path", type=str, default="ground_truth_data_info/",
                        help="directory to store core data information.")

    parser.add_argument("--train_file", type=str,
                        default="../data/AGNews/train_data.json",
                        help="training set")
    parser.add_argument("--dev_file", type=str,
                        default="../data/AGNews/dev_data.json", help="dev set")
    parser.add_argument("--test_file", type=str,
                        default="../data/AGNews/test_data.json", help="test set")
    parser.add_argument("--unlabeled_file", type=str,
                        default="../data/AGNews/unlabeled_data.json", help="unlabeled set")

    parser.add_argument("--data_selection", type=bool, default=True, help="use selected data or all data.")
    parser.add_argument("--num_labels", type=int, default=4000, help="the number of labeled data.")
    parser.add_argument("--init_val", type=bool, default=False, help="reinit val data.")
    parser.add_argument("--num_val", type=int, default=4000, help="the numer of val data.")
    parser.add_argument("--data_strategy", type=str, default="random", help="the strategy to select data.")
    parser.add_argument("--ratio_noisy_labels", type=float, default=0.2, help="the strategy to select data.")
    parser.add_argument("--clean", action='store_true', help="train with noisy label of clean label")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--no_cuda", type=bool, default=False, help="GPU or not")

    parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--unl_batch_size", type=int, default=256, help="size of the batches on unlabeled data")

    parser.add_argument("--lr", type=float, default=1e-5, help="sgd: learning rate")
    parser.add_argument("--momentum", type=float, default=0.5, help="sgd: momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="sgd: weight_decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="adam: beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="adam: beta2")
    parser.add_argument("--max_scale", type=float, default=100, help="max scale of unsupervised loss")
    parser.add_argument("--T", type=float, default=0.5, help="ces: temperature")

    parser.add_argument("--hidden_size", type=int, default=768, help="the number of hidden size in bert")

    args = parser.parse_args()
    print(args)

    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print("Set seed :", seed)

    # if args.data_text:
    trainer = Trainer_text(args)
    # else:
    # trainer = Trainer(args)
    trainer.main()