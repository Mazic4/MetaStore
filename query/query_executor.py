import os
import timeit

from ted_v3.artifacts_calculator.artifacts_collector import _get_data_artifacts
from ted_v3.core_engine.meta_grad_calculator import _get_meta_gradients

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def query(analyzer, test_artifacts_list, method="naive"):

    meta_gradient = _get_meta_gradients(analyzer, test_artifacts_list, method=method, log_name=method)

    return meta_gradient