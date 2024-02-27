# MetaStore: Analyzing Deep Learning Meta-Data at Scale

## 🔔 News
- **[02/09/2024]** Our code is open-sourced! The code repo is still under construction, Please contact zhanghuayi01@gmail.com if you have any questions.

We support the following methods: 
* naive: The naive method that store and compute on full gradient
* recon: A p2b method that use \<prefix, suffix\> pairs to store/load gradients, but reconstruct gradients in GPU.
* ted: the method that run with p2p operators in MetaStore 
* half: the method that run with p2b operators in MetaStore

## Setup
The following steps provide the necessary setup to run our codes.
1. Create a Python virtual environment with Conda:
```
conda create -n myenv python=3.7
conda activate myenv
```
2. Install PyTorch with compatible cuda version, following instructions from [PyTorch Installation Page](https://pytorch.org/get-started/locally/). For example with cuda 11:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
3. Install the following Python dependencies to run the codes.
```
pip install -r requirements.txt
```
## Usage Example
Using the CIFAR10/AGNews/Imagenet dataset as example, you can test MetaStore with the following methods:

You can run the scripts to test MetaStore:
```
bash ./scripts/experiment_run.sh
```

## Instructions to apply MetaStore on new Models/Datasets:

You can use the following code to obtain the results:
### Step 0-1: Train a Model and store the checkpoints
```
cd ./metastore
python3 ./model_trainers/vgg16_trainer.py
```
### Step 0-2: Obtain the target layer names with the ```unwarp``` function. For example,
```
from utils.utils import unwrap
from model_trainers.models import VGG16
import torch.nn as nn

model = VGG16()
model.load_state_dict(torch.load("path/to/your/model")))
valid_layer_types = []
valid_layer_types.append(nn.Linear)
valid_layer_types.append(nn.Conv2d)
flatten_layers = unwrap(model, "your_model_name", valid_layer_types)
print (flatten_layers)
```
then add the target layers into the config files, e.g., config.yaml

### Step 1: Collect and store the gradient with Model Reply
Set ```config.system.collect_artifacts=true``` in the config file will collect and store the gradients.

### Step 2: Run queries, e.g. p2b operators and p2p operators, for example,
```
CUDA_VISIBLE_DEVICES=${device_number} python3 ./main.py --config ./config.yaml --method ted --target_model VGG16 \
  --data cifar10 --num_analyzed_samples 10000 --num_query 100
```
You can also formulate the test samples by set the indices of testing samples in the main.py file,
```
test_loader, test_dataset = get_dataloader_cifar(opt["data"], mode="test", indices="INDICES_OF_TEST_SAMPLES"))
```


## Tips
Some important hyper-parameters to tune given a new dataset to avoid OOM:
1. gpu_memory_cache_threshold
2. store_batch_size

### Contact
Please contact zhanghuayi01@gmail.com if you have any questions.