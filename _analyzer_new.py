import os

import torch.nn as nn

from model_trainers.models import *
from artifacts_storage.artifacts_store import store_traindata_artifacts

from utils import *
from utils import data_loader_new as dataloader
from utils.utils import *
from utils.unwrap import  unwrap
from utils.grad_to_feats import *


import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Analyzer:

    def __init__(self, args)->None:

        self.args = args

        self.set_path()
        self.set_variable()
        self.set_model()
        self.set_traindata()
        self.set_quant()
        self.get_gpu_size()
        self.artifacts_size_known=self.get_artifacts_size()
        if not self.artifacts_size_known:
            store_traindata_artifacts(self, method=self.args["system"]["method"])
        self.artifacts_size_known = self.get_artifacts_size()

    def set_variable(self):

        self.method = self.args["system"]["method"]
        self.dataset_name = self.args["data"]["data_name"]
        self.max_store_batch_size = self.args["system"]["io"]["max_store_batch_size"]

        # check method
        system_config = self.args["system"]

        self.valid_methods = system_config["valid_methods"]
        if system_config["method"] not in self.valid_methods:
            raise NotImplementedError("Method {} is not found.".format(self.args.method))

        self.valid_layer_types_str = system_config["valid_layer_types"]
        self.valid_layer_types = []
        for layer_type_str in self.valid_layer_types_str:
            if layer_type_str == "linear":
                self.valid_layer_types.append(nn.Linear)
            elif layer_type_str == "conv2d":
                self.valid_layer_types.append(nn.Conv2d)
            else:
                raise NotImplementedError(layer_type_str)
        # todo: add a layer type valid func

        # check model config
        model_config = self.args["target_model"]
        self.target_layers = model_config["target_layers"]

        #others
        self.cached_batches = []
        self.cached_train_samples_idx = []

        self.train_ind_grad_dict = {key: [] for key in self.target_layers}

        self.train_dy_dw_dict = {key: [] for key in self.target_layers}
        self.train_dl_dy_dict = {key: [] for key in self.target_layers}

        self.default_dtype = torch.float32
        torch.set_default_dtype(torch.float32)


    def set_path(self):

        #artifacts storage
        artifacts_store_base = self.args["system"]["artifacts_store_base"]
        artifacts_store_data = "{}_artifacts_layer".format(self.args["data"]["data_name"])
        self.artifacts_log_path = os.path.join(artifacts_store_base, artifacts_store_data)

        if self.args["system"]["quant"]["use_quant"]:
            float_precision = self.args["system"]["quant"]["quant_float_precision"]
            suffix_quant_artifacts_log = "_quant"+"_{}".format(float_precision)
            self.artifacts_log_path = os.path.join(self.artifacts_log_path, suffix_quant_artifacts_log)

        self._artifacts_log_path = os.path.join(artifacts_store_base, "_cache")

        #set output_path
        system_config = self.args["system"]
        self.output_path = os.path.join(system_config["out_dir_base"], system_config["out_result_dir"])
        self.output_meta_path = os.path.join(system_config["out_dir_base"], system_config["out_meta_result_dir"])

        os.makedirs(self.artifacts_log_path, exist_ok=True)
        os.makedirs(self._artifacts_log_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.output_meta_path, exist_ok=True)


    def set_quant(self):
        quant_args = self.args["system"]["quant"]
        self.use_quant = quant_args["use_quant"]
        self.default_quant_type = None
        if quant_args["use_quant"]:
            float_precision = quant_args["quant_float_precision"]
            if float_precision == "torch.quint8":
                self.default_quant_type = torch.quint8
            elif float_precision == "torch.qint8":
                self.default_quant_type = torch.qint8
            elif float_precision == "torch.qint32":
                self.default_quant_type = torch.qint32
            elif float_precision == "torch.float16":
                self.default_quant_type = torch.float16
            else:
                raise NotImplementedError


    def set_traindata(self):
        data_args = self.args['data']
        # load the training data samples
        train_indices = np.arange(data_args["num_analyzed_samples"])
        if data_args["data_name"] == "cifar10":
            self.train_loader, self.train_dataset  = dataloader.get_dataloader_cifar(data_args, mode='train',
                                                                                    indices=train_indices)
        elif data_args["data_name"] == "AGNews":
            self.train_loader, self.train_dataset = dataloader.get_dataloader_agnews(data_args, mode='train',
                                                                                    indices=train_indices)
        else:
            raise NotImplementedError

        self.num_batches = len(self.train_dataset) // self.args["system"]["io"]["max_store_batch_size"]


    def set_model(self):
        model_args = self.args['target_model']

        if model_args["model_name"] == "VGG16":
            self.model = VGG16()
            self.model.load_state_dict(
                torch.load('./models/vgg16_models/cifar10_state_dict_finetune_{}.pth'.
                           format(model_args["target_epoch"])))

        elif model_args["model_name"] == "Bert":
            self.model = Customized_Bert_Model(num_classes=4).cuda()
            self.model.load_state_dict(
                torch.load('./models/bert_models/agnews_state_dict_finetune_{}_{}.pth'.
                           format(model_args["target_epoch"], 768)), strict=False)
            #todo: clean the model path
        else:
            raise NotImplementedError

        self.model.eval().cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=model_args["lr"], momentum=model_args["momentum"])
        self.loss_func = nn.CrossEntropyLoss()

        self.flatten_model = unwrap(self.model, "net", self.valid_layer_types)


    def get_gpu_size(self):
        self.gpu_max_size = get_gpu_memory() * self.args["system"]["io"]["gpu_memory_threshold"]


    def get_artifacts_size(self):
        self.artifacts_size = {}
        if self.method in ["ted", "half", "recon", "iter"]:
            batch_idx = 0
            for layer_name in self.target_layers:
                file_name_1 = "{}/{}_batch_{}_dy_dw.pt".format(self.artifacts_log_path, layer_name, batch_idx)
                file_name_2 = "{}/{}_batch_{}_dl_dy.pt".format(self.artifacts_log_path, layer_name, batch_idx)
                if not os.path.exists(file_name_1):
                    return False

                total_file_size_1 = os.path.getsize(file_name_1) / 1e6
                total_file_size_2 = os.path.getsize(file_name_2) / 1e6
                self.artifacts_size[layer_name] = {"ted_input_train": total_file_size_1,
                                                   "ted_output_grad_train": total_file_size_2}
        elif self.method in ["naive"]:
            batch_idx = 0
            for layer_name in self.target_layers:
                file_name = "{}/{}_batch_{}_ind_grad.pt".format(self.artifacts_log_path, layer_name, batch_idx)
                if not os.path.exists(file_name):
                    return False
                total_file_size_1 = os.path.getsize(file_name) / 1e6
                self.artifacts_size[layer_name] = {"naive": total_file_size_1}
        else:
            raise NotImplementedError



