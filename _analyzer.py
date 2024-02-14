import os

import torch.nn as nn

from utils import *
from utils.utils import *
from utils.unwrap import  unwrap
from utils.grad_to_feats import *


import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Analyzer():
    def __init__(self, args)->None:

        self.args = args

        #check method
        self.valid_methods = ["naive", "ted", "half", "recon"]
        if self.args.method not in self.valid_methods:
            raise NotImplementedError("Method {} is not found.".format(self.args.method))
        print ("Method is ", self.args.method)

        #check layer type
        self.valid_layers = [nn.Linear, nn.Conv2d]

        if self.args.dataset == "AGNews":
            if self.args.experiment_type in ["model", "batch"]:
                self.target_layers = ['net_bert_encoder_layer_11_attention_self_query',
                                      'net_bert_encoder_layer_11_attention_self_key',
                                      'net_bert_encoder_layer_11_attention_self_value',
                                      'net_bert_encoder_layer_6_attention_self_query',
                                      'net_bert_encoder_layer_6_attention_self_key',
                                      'net_bert_encoder_layer_6_attention_self_value',
                                      'net_bert_encoder_layer_1_attention_self_query',
                                      'net_bert_encoder_layer_1_attention_self_key',
                                      'net_bert_encoder_layer_1_attention_self_value'
                                      ]
            else:
                self.target_layers = ['net_bert_encoder_layer_11_attention_self_query',
                                      'net_bert_encoder_layer_11_attention_self_key',
                                      'net_bert_encoder_layer_11_attention_self_value']

        elif self.args.dataset == "cifar10":
            if self.args.experiment_type in ["model", "batch"]:
                self.target_layers = ["net_conv1", "net_conv7", "net_conv13", "net_l1"]
            else:
                self.target_layers = ["net_conv_append_layer", "net_linear_append_layer"]

        elif self.args.dataset == "imagenet":
            if self.args.experiment_type in ["model", "batch"]:
                self.target_layers = ["net_l1", "net_model_base_layer4_2_conv2"]
            else:
                self.target_layers = ["net_linear_append_layer"]

        self.artifacts_log_path = "./artifacts/{}_artifacts_layer".format(self.args.dataset)
        os.makedirs(self.artifacts_log_path, exist_ok=True)

        if self.args.precision_mode == "quant":
            self.artifacts_log_path += "_quant"
            self.artifacts_log_path += "_{}".format(self.args.float_precision)

        if self.args.precision_mode == "quant":
            if self.args.float_precision == "torch.quint8":
                self.default_quant_type = torch.quint8
            elif self.args.float_precision == "torch.qint8":
                self.default_quant_type = torch.qint8
            elif self.args.float_precision == "torch.qint32":
                self.default_quant_type = torch.qint32
            elif self.args.float_precision == "torch.float16":
                self.default_quant_type = torch.float16
        else:
            self.default_quant_type = None

        self.default_dtype = torch.float32
        torch.set_default_dtype(torch.float32)

        if not os.path.exists(self.args.output_path):
            os.mkdir(self.args.output_path)

        self.output_path = self.args.output_path + \
                           "/{}".format(self.args.experiment_type) + \
                           "_{}".format(self.args.dataset) + \
                           "_{}".format(self.args.num_samples) + \
                           "_{}".format(self.args.num_query)

        if self.args.experiment_type == "layer":
            self.output_path += "_{}".format(self.args.analyze_layer_type)
            self.output_path += "_{}".format(self.args.hidden_size)
        elif self.args.experiment_type == "batch":
            self.output_path += "_{}".format(self.args.num_pseudo_samples)

        self._artifacts_log_path = "./artifacts/_cache"

        if not os.path.exists(self.artifacts_log_path):
            os.mkdir(self.artifacts_log_path)
        if not os.path.exists(self.args.output_path):
            os.mkdir(self.args.output_path)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        if not os.path.exists(self._artifacts_log_path):
            os.mkdir(self._artifacts_log_path)

        self.visited_checkpoints_naive = set([])
        self.visited_checkpoints_ted = set([])

        print ("Target layers include:", self.target_layers)

        self.init_memory_manager()

    def init_memory_manager(self):

        self.cached_batches = []
        self.cached_train_samples_idx = []

        self.train_ind_grad_dict = {key: [] for key in self.target_layers}

        self.train_dy_dw_dict = {key: [] for key in self.target_layers}
        self.train_dl_dy_dict = {key: [] for key in self.target_layers}

    def get_gpu_size(self):
        self.gpu_max_size = get_gpu_memory() * self.args.gpu_memory_threshold

    def get_artifacts_size(self):

        self.artifacts_size = {"ted_input_train":{}, "ted_output_grad_train":{}, "naive":{}}
        batch_idx = 0
        total_file_size_1, total_file_size_2 = 0, 0
        for layer_name in self.target_layers:
            total_file_size_1 += os.path.getsize("{}/{}_batch_{}_dy_dw.pt"
                                                 .format(self.artifacts_log_path, layer_name, batch_idx)) / 1e6
            total_file_size_2 += os.path.getsize("{}/{}_batch_{}_dl_dy.pt"
                                                 .format(self.artifacts_log_path, layer_name, batch_idx)) / 1e6
            self.artifacts_size["ted_input_train"][layer_name] = total_file_size_1
            self.artifacts_size["ted_output_grad_train"][layer_name] = total_file_size_2

        batch_idx = 0
        total_file_size_1, total_file_size_2 = 0, 0
        for layer_name in self.target_layers:
            total_file_size_1 += os.path.getsize("{}/{}_batch_{}_ind_grad.pt"
                                                 .format(self.artifacts_log_path, layer_name, batch_idx)) / 1e6
            self.artifacts_size["naive"][layer_name] = total_file_size_1

    def load_models(self, model, optimizer)->None:

        self.model = model
        self.optimizer = optimizer

        self.model.eval()

        self.flatten_model = unwrap(self.model, "net", self.valid_layers)

    def set_loss_func(self, loss_func)->None:
        self.loss_func = loss_func

    def set_traindata(self, dataset)->None:
        #load the training data samples
        self.train_dataset = dataset
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.args.batch_size, shuffle=False)
        self.num_batches = len(self.train_dataset)//self.args.max_store_batch_size
