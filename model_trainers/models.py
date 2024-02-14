import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision

from transformers import AdamW, AutoTokenizer, BertConfig, get_linear_schedule_with_warmup, AutoModel, BertModel
from transformers.modeling_outputs import  SequenceClassifierOutput
from transformers import BertForSequenceClassification

from utils.unwrap import *


class Network(nn.Module):

    def __init__(self, hidden_size=50):
        super(Network, self).__init__()

        # self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l0 = nn.Linear(28*28, 512)
        self.relu0 = nn.ReLU()

        self.l1 = nn.Linear(512, hidden_size)
        self.relu1 = nn.ReLU()

        self.l2 = nn.Linear(hidden_size, 10)

    def forward(self, x):

        if len(x.shape) < 4:
            x = x.unsqueeze(0)

        x = x.reshape(-1, 28*28)

        x = self.l0(x)
        x = self.relu0(x)

        x = self.l1(x)
        x = self.relu1(x)

        x = self.l2(x)

        return x


class Network1(nn.Module):
    def __init__(self) -> None:
        super(Network1, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x

# todo: fix the lenet


class LeNet(nn.Module):

    def __init__(self, hidden_size=50):
        super(LeNet, self).__init__()

        self.conv2d_1 = nn.Conv2d(
            in_channels=3, out_channels=512, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()

        self.conv2d_2 = nn.Conv2d(
            in_channels=512, out_channels=hidden_size, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()

        self.l3 = nn.Linear(hidden_size, 10)

    def forward(self, x):

        if len(x.shape) < 4:
            x = x.unsqueeze(0)

        x = x.reshape(-1, 3*28*28)

        x = self.l0(x)
        x = self.relu0(x)

        x = self.l1(x)
        x = self.relu1(x)

        x = self.l2(x)

        return x


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.relu10 = nn.ReLU()
        self.pool10 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.relu12 = nn.ReLU()
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.relu13 = nn.ReLU()
        self.pool13 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.append_layer_type = None
        self.l1 = nn.Linear(512, 10)

    def append(self, layer_type="linear", hidden_size=512):

        self.append_layer_type = layer_type

        if layer_type == "linear":
            self.linear_append_layer = nn.Linear(512, hidden_size)
            self.relu_append = nn.ReLU()
            self.l1 = nn.Linear(hidden_size, 10)
        elif layer_type == "conv":
            self.conv_append_layer = nn.Conv2d(
                512, hidden_size, kernel_size=3, padding=1)
            self.batchnorm_append = nn.BatchNorm2d(hidden_size)
            self.relu_append = nn.ReLU()
            self.l1 = nn.Linear(hidden_size, 10)
        else:
            raise ValueError(
                "The append layer type {} is not implemented yet.".format(layer_type))

    def forward(self, x, feature=False):

        if len(x.shape) < 4:
            x = x.unsqueeze(0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.pool7(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu10(x)
        x = self.pool10(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu12(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu13(x)
        x = self.pool13(x)

        if feature:
            return x

        if self.append_layer_type is not None:

            if self.append_layer_type == "linear":

                l = x.shape[0]
                x = x.reshape(l, -1, )
                x = self.linear_append_layer(x)
                x = self.relu_append(x)

            elif self.append_layer_type == "conv":

                x = self.conv_append_layer(x)
                x = self.batchnorm_append(x)
                x = self.relu_append(x)
                l = x.shape[0]
                x = x.reshape(l, -1, )

            else:
                raise ValueError("The append layer type {} is not implemented yet.".format(
                    self.append_layer_type))
        # print(x.shape)
        x = torch.squeeze(x, -1)
        x = torch.squeeze(x, -1)
        x = self.l1(x)

        return x


class Customized_Bert_Model(nn.Module):

    def __init__(self, hidden_size = 768, num_classes=4):

        super(Customized_Bert_Model, self).__init__()

        # Initializing a BERT bert-base-uncased style configuration
        configuration = BertConfig(hidden_size = hidden_size)

        # Initializing a model from the bert-base-uncased style configuration
        self.bert = BertModel(configuration)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, **inputs):

        hidden_state = self.bert(**inputs)[0][:, 0, :]
        logits = self.classifier(hidden_state)

        return SequenceClassifierOutput(logits=logits)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResNet50(nn.Module):

    def __init__(self):

        super(ResNet50, self).__init__()

        resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

        layers = unwrap(resnet, name="net", valid_layer=[nn.Linear, nn.Conv2d])

        # for name, layer in layers:
        #     if name in ["net_l1", "net_layer4_2_conv2", "net_model_base_layer4_2_conv2", "net_linear_append_layer"]:
        #         continue
        #     else:
        #         for param in layer.parameters():
        #             param.requires_grad = False

        # for name, child in resnet.named_children():
            # if name == "fc":
            #     continue
            #
            # elif name == "layer4":
            #     for _name, _child in child.named_children():
            #         if _name == ""
            #
            # else:
            #     for param in child.parameters():
            #         param.requires_grad = False


        resnet.fc = Identity()
        self.append_layer_type = None
        self.l1 = nn.Linear(2048, 1000)

        # resnet.fc = nn.Linear(2048, 1000)  # Reinitializes final layer, assigns random weights

        self.model_base = resnet


    def append(self, layer_type="linear", hidden_size=1024):

        self.append_layer_type = layer_type

        if layer_type == "linear":
            self.linear_append_layer = nn.Linear(2048, hidden_size)
            self.relu_append = nn.ReLU()
            self.l1 = nn.Linear(hidden_size, 1000)
        else:
            raise ValueError(
                "The append layer type {} is not implemented yet.".format(layer_type))


    def forward(self, x, feature=False):

        x = self.model_base(x)

        if feature:
            return x

        else:
            if self.append_layer_type is not None:
                l = x.shape[0]
                x = x.reshape(l, -1)
                x = self.relu_append(self.linear_append_layer(x))
                x = self.l1(x)
            else:
                l = x.shape[0]
                x = x.reshape(l, -1)
                x = self.l1(x)

        return x

