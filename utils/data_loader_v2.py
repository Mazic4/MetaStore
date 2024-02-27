import os

import numpy as np

import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer

import pandas as pd
from .utils import pad_sequences


class Custom_Dataset():
    def __init__(self, dataset, index = None):
        self.data = dataset
        if index is None:
            self.index = np.arange(len(dataset))
        else:
            self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.index[idx], self.data[self.index[idx]][0], self.data[self.index[idx]][1]

class Custom_Dataset_text(Dataset):
    def __init__(self, all_input_ids_new, all_attention_mask_new, all_label, all_true_label=None, index=None):

        self.input_ids = all_input_ids_new
        self.attention_masks = all_attention_mask_new
        self.labels = all_label
        self.true_labels = all_true_label

        if index is None:
            self.index = np.arange(len(self.input_ids))
        else:
            self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        _idx = self.index[idx]

        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        label = self.labels[idx]
        true_label = self.true_labels[idx]

        return _idx, {'input_ids': input_id, 'attention_mask': attention_mask}, true_label

def process_dataset(args, mode, tokenizer, indices = None, max_length=32):

    if mode == 'train':
        d1 = pd.read_json(args["train_file"], orient='records')

    elif mode == 'dev':
        d1 = pd.read_json(args["dev_file"], orient='records')

    elif mode == 'test':
        d1 = pd.read_json(args["test_file"], orient='records')

    elif mode == 'unlabeled':
        d1 = pd.read_json(args["unlabeled_file"], orient='records')

    if indices is None:
        indices = np.arange(len(d1))

    encoded_input = tokenizer(d1['text'].tolist(), padding=True, truncation=True, return_tensors="pt")

    all_input_ids = encoded_input['input_ids']
    all_attention_mask = encoded_input['attention_mask']

    # padding to max_length
    all_input_ids_new = torch.tensor(
        pad_sequences(all_input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post"))
    #     all_token_type_ids_new = pad_sequences(all_token_type_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    all_attention_mask_new = torch.tensor(
        pad_sequences(all_attention_mask, maxlen=max_length, dtype="long", truncating="post", padding="post"))

    all_label = torch.tensor(d1['label'])

    all_true_label = torch.tensor(d1['label'])
    dataset = Custom_Dataset_text(all_input_ids_new, all_attention_mask_new, all_label, all_true_label, index = indices)

    return dataset

def get_dataloader_cifar(args, mode, indices):

    if mode == "train":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([32, 32]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        dataset = Custom_Dataset(
            torchvision.datasets.CIFAR10('data', transform=transform, train=True, download=True),
            index=indices)

        loader = torch.utils.data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)

    elif mode == "test":

        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize([32, 32]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataset = Custom_Dataset(
            torchvision.datasets.CIFAR10('data', transform=transform_test, train=False, download=True),
            index=indices)

        loader = torch.utils.data.DataLoader(dataset,batch_size=args["batch_size"], shuffle=True)

    else:
        raise NotImplementedError

    return loader, dataset

def get_dataloader_agnews(args, mode, indices):

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = process_dataset(args, mode, tokenizer, indices)
    data_loader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=False)
    return data_loader, dataset


def get_dataloader_imagenet(args, mode, indices):

    data_dir = args["data_dir"]

    if mode == "train":
        transform = transforms.Compose([
            transforms.Resize(64),  # Takes images smaller than 64 and enlarges them
            transforms.RandomCrop(64, padding=4, padding_mode='edge'),  # Take 64x64 crops from 72x72 padded images
            transforms.RandomHorizontalFlip(),  # 50% of time flip image along y-axis
            transforms.ToTensor(),
        ])

        dataset = Custom_Dataset(
            torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform),
            index=indices)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=True, num_workers=2)

    elif mode == "test":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = Custom_Dataset(
            torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform),
            index=indices)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=False, num_workers=2)
    else:
        raise NotImplementedError


    return loader, dataset