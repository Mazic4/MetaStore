import os
import timeit

from ted_v3.core_engine.meta_grad_calculator import _get_meta_gradients


import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def query_batch(analyzer, test_grad_list, method="naive"):

    meta_gradient = _get_meta_gradients(analyzer, test_grad_list, method=method)

    return meta_gradient