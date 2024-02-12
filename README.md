We support the following methods: 
* Naive: directly compute, store, load and compute on gradients.
* p2p: P2P operator of MetaStore
* p2b: P2B operator of MetaStore

Using the CIFAR10 dataset as example, you can test MetaStore with the following code:
* Naive:
* p2p:
* p2b:

You can use the following code to obtain the results:


Instructions to test on new models/layers:
1. train a model with the dataset and model
2. store the checkpoints
3. obtain the layer name of the new model
4. add the layer name to the target layers in the Analyzer class


Some important hyper-parameters to tune given a new dataset:
1. gpu_memory_cache_threshold
2. store_batch_size

Requirements:
