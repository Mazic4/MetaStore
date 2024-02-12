import os

import numpy as np

import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer

import pandas as pd
from .utils import pad_sequences


class Customized_Dataset():
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

class Customized_Dataset_noisy():
    def __init__(self, dataset, labels, index = None):
        self.data = dataset
        self.labels = labels

        if index is None:
            self.index = np.arange(len(dataset))
        else:
            self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.index[idx], self.data[self.index[idx]][0], self.labels[idx]


class Customizer_Dataset_text(Dataset):
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

class Customizer_Dataset_noisy_text(Dataset):
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

        return _idx, {'input_ids': input_id, 'attention_mask': attention_mask}, label, true_label

class Customizer_Dataset_feature(Dataset):
    def __init__(self, all_features, all_label, all_true_label=None, index=None):

        self.feature = all_features
        self.labels = all_label
        self.true_labels = all_true_label

        if index is None:
            self.index = np.arange(len(self.feature))
        else:
            self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        _idx = self.index[idx]

        feature = self.feature[idx]
        label = self.labels[idx]
        true_label = self.true_labels[idx]

        return feature, label, true_label

def get_dataset(args):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    _train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    _test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)

    _index = np.arange(50000)
    np.random.shuffle(_index)
    clean_lab_idx = _index[:int((1 - args.ratio_noisy_labels) * args.num_labels)]
    noisy_lab_idx = _index[int((1 - args.ratio_noisy_labels) * args.num_labels): args.num_labels]
    lab_idx = _index[:args.num_labels]
    unl_idx = _index[args.num_labels:]

    train_labels = []
    for i in range(len(_train_dataset)):
        org_label = _train_dataset[i][1]
        if i in noisy_lab_idx:
            noisy_label = np.random.choice(np.arange(10), size=1, replace=False)[0]
            while noisy_label == org_label:
                noisy_label = np.random.choice(np.arange(10), size=1, replace=False)[0]
            train_labels.append(noisy_label)
        else:
            train_labels.append(org_label)

    train_labels = np.array(train_labels)
    # np.save("lab_data_index_{}.npy".format(args.ratio_noisy_labels), lab_idx)
    # np.save("train_labels_{}.npy".format(args.ratio_noisy_labels), train_labels)

    lab_dataset = Customizer_Dataset(_train_dataset, train_transform, train_labels, lab_idx)
    unl_dataset = Customizer_Dataset(_train_dataset, train_transform, train_labels, unl_idx)
    test_dataset = Customizer_Dataset(_test_dataset, test_transform)

    return lab_dataset, unl_dataset, test_dataset

def Process_dataset(args, mode, tokenizer, indices = None, max_length=32):

    if mode == 'train':
        d1 = pd.read_json(args.train_file, orient='records')

    elif mode == 'dev':
        d1 = pd.read_json(args.dev_file, orient='records')

    elif mode == 'test':
        d1 = pd.read_json(args.test_file, orient='records')

    elif mode == 'unlabeled':
        d1 = pd.read_json(args.unlabeled_file, orient='records')

    if indices is None:
        indices = np.arange(len(d1))

    encoded_input = tokenizer(d1['text'].tolist(), padding=True, truncation=True, return_tensors="pt")

    all_input_ids = encoded_input['input_ids']
    #     all_token_type_ids = encoded_input['token_type_ids']
    all_attention_mask = encoded_input['attention_mask']

    # padding to max_length
    all_input_ids_new = torch.tensor(
        pad_sequences(all_input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post"))
    #     all_token_type_ids_new = pad_sequences(all_token_type_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    all_attention_mask_new = torch.tensor(
        pad_sequences(all_attention_mask, maxlen=max_length, dtype="long", truncating="post", padding="post"))

    all_label = torch.tensor(d1['label'])

    all_true_label = torch.tensor(d1['label'])
    dataset = Customizer_Dataset_text(all_input_ids_new, all_attention_mask_new, all_label, all_true_label, index = indices)

    return dataset

def Process_dataset_noisy(args, mode, tokenizer, indices = None, max_length=32):

    if mode == 'train':
        d1 = pd.read_json(args.train_file, orient='records')

    elif mode == 'dev':
        d1 = pd.read_json(args.dev_file, orient='records')

    elif mode == 'test':
        d1 = pd.read_json(args.test_file, orient='records')

    elif mode == 'unlabeled':
        d1 = pd.read_json(args.unlabeled_file, orient='records')

    if indices is None:
        indices = np.arange(len(d1))

    encoded_input = tokenizer(d1['text'].tolist(), padding=True, truncation=True, return_tensors="pt")

    all_input_ids = encoded_input['input_ids']
    #     all_token_type_ids = encoded_input['token_type_ids']
    all_attention_mask = encoded_input['attention_mask']

    # padding to max_length
    all_input_ids_new = torch.tensor(
        pad_sequences(all_input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post"))
    #     all_token_type_ids_new = pad_sequences(all_token_type_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    all_attention_mask_new = torch.tensor(
        pad_sequences(all_attention_mask, maxlen=max_length, dtype="long", truncating="post", padding="post"))

    all_label = torch.tensor(d1['label']).cpu().numpy()

    idx = np.arange(len(indices))
    np.random.shuffle(idx)
    noisy_idx = idx[:int(len(idx) * 0.2)]
    for i in noisy_idx:
        org_label = all_label[i]
        noisy_label = np.random.randint(low=0, high=4)
        while noisy_label == org_label:
            noisy_label = np.random.randint(low=0, high=4)
        all_label[i] = noisy_label

    all_label = np.array(all_label)
    with open('./models/bert_models/agnews_noisy_labels_0.2.npy', 'wb') as f:
        np.save(f, all_label)

    all_label = torch.from_numpy(all_label)
    all_true_label = torch.tensor(d1['label'])

    dataset = Customizer_Dataset_noisy_text(all_input_ids_new, all_attention_mask_new, all_label, all_true_label, index = indices)

    return dataset

def get_dataset_text(args, tokenizer, train_indices = None, test_indices = None):
    l_dataset = Process_dataset(args, 'train', tokenizer, train_indices)
    unl_dataset = Process_dataset(args, 'unlabeled', tokenizer)
    test_dataset = Process_dataset(args, 'test', tokenizer, test_indices)
    dev_dataset = Process_dataset(args, 'dev', tokenizer)

    return l_dataset, unl_dataset, test_dataset, dev_dataset

def get_dataset_text_noisy(args, tokenizer, train_indices = None, test_indices = None):
    l_dataset = Process_dataset_noisy(args, 'train', tokenizer, train_indices)
    unl_dataset = Process_dataset(args, 'unlabeled', tokenizer)
    test_dataset = Process_dataset(args, 'test', tokenizer, test_indices)
    dev_dataset = Process_dataset(args, 'dev', tokenizer)

    return l_dataset, unl_dataset, test_dataset, dev_dataset

def get_dataset_features(args):
    data_file = torch.load(args.train_file)

    l_dataset = data_file['labeled']
    unl_dataset = data_file['unlabeled']
    test_dataset = data_file['test']
    dev_dataset = data_file['validation']

    print('acc rate of major labels:', sum(l_dataset['major_label'] == l_dataset['label']) / len(l_dataset['label']))

    lab_dataset = Customizer_Dataset_feature(l_dataset['bert_feature'], l_dataset['major_label'].long(),
                                             l_dataset['label'].long())
    unl_dataset = Customizer_Dataset_feature(unl_dataset['bert_feature'], unl_dataset['major_label'].long(),
                                             unl_dataset['label'].long())
    dev_dataset = Customizer_Dataset_feature(dev_dataset['bert_feature'], dev_dataset['major_label'].long(),
                                             dev_dataset['label'].long())
    test_dataset = Customizer_Dataset_feature(test_dataset['bert_feature'], test_dataset['major_label'].long(),
                                              test_dataset['label'].long())

    return lab_dataset, unl_dataset, test_dataset, dev_dataset

def get_dataloaders(args, dataset = "cifar10", train_indices = None, test_indices = None, noisy_ratio = None):
    
    if dataset == "cifar10":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([32, 32]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize([32, 32]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if noisy_ratio is None:
            train_dataset = Customized_Dataset(
                torchvision.datasets.CIFAR10('data', transform=transform, train=True, download=True), index=train_indices)
            test_dataset = Customized_Dataset(
                torchvision.datasets.CIFAR10('data', transform=transform_test, train=False, download=True), index=test_indices)
        else:
            noisy_labels = np.load("./model_trainers/models/vgg16_models/cifar_noisy_labels_{}.npy".format(noisy_ratio))
            train_dataset = Customized_Dataset_noisy(
                torchvision.datasets.CIFAR10('data', transform=transform, train=True, download=True), noisy_labels,
                index=train_indices)
            test_dataset = Customized_Dataset(
                torchvision.datasets.CIFAR10('data', transform=transform_test, train=False, download=True),
                index=test_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=args.batch_size, shuffle=True)

    elif dataset =="AGNews":

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        train_dataset, unlabeled, test_dataset, dev_dataset = get_dataset_text(args, tokenizer, train_indices, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    elif dataset == "imagenet":

        data_dir = './model_trainers/data/'

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

        train_dataset = Customized_Dataset(
            torchvision.datasets.ImageFolder(root= data_dir + '/imagenet64/train/', transform=transform_train), index=train_indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        test_dataset = Customized_Dataset(
            torchvision.datasets.ImageFolder(root= data_dir + '/imagenet64/val/', transform=transform_test), index=test_indices)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    else:
        print("Using MNIST as dataset")
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])

        train_dataset = Customized_Dataset(
            torchvision.datasets.MNIST('/dataset/', train=True, download=True, transform=transform))

        test_dataset = Customized_Dataset(
            torchvision.datasets.MNIST('/dataset/', train=False, download=True, transform=transform))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=args.batch_size, shuffle=True)


    return train_loader, test_loader, train_dataset, test_dataset