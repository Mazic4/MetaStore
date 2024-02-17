import os
import subprocess as sp
import timeit
from functools import wraps

from typing import Dict

import six

import numpy as np

import torch
from torch.autograd import Variable

def inner_product_batch(t1_single, t2_large, _sum_axis, batch_size = 100):
    """
    This is used to calculate inner product batch by batch
    :param t1: a single tensor
    :param t2: a set of tensors
    :return:
    """

    res = []

    total_batch = t2_large.shape[0]//batch_size + 1
    for i in range(total_batch):
        _t2 = t2_large[i*batch_size : (i+1)*batch_size]
        _accum = torch.multiply(t1_single, _t2).sum(axis=_sum_axis)
        res.append(_accum)

    res = torch.cat(res, dim=0)

    return res

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.
    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.
    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.
    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding is the default.
    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.
    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`
    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def get_files(path, key_words = []):

    all_artifacts_files = os.listdir(path)

    target_files = []

    for file in all_artifacts_files:
        if all([key_word in file for key_word in key_words]):
            target_files.append(file)

    return target_files

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

    return memory_free_values[0]
    # return torch.cuda.get_device_properties(0).total_memory * 0.01

def svd(matrix, num_components):

    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    U = U[:, :num_components]
    S = S[:num_components]
    Vh = Vh[:num_components]

    return U, S, Vh

def verify_reconstructed_metagradinet(ground_truth_meta_gradinet, reconstructed_meta_gradinet)->bool:
    return np.allclose(ground_truth_meta_gradinet, reconstructed_meta_gradinet, atol=1e-6)


def detect_noisy_sample_baselines(feats, loss = None, method="lof"):

    from sklearn.neighbors import LocalOutlierFactor

    lof = LocalOutlierFactor()

    return lof.fit_predict(feats)


def timer(runtime_var: Dict[str, float]):
    def _timer(func):
        @wraps(func)
        def timeit_wrapper(*args, **kwargs):
            start_time = timeit.default_timer()
            result = func(*args, **kwargs)
            end_time = timeit.default_timer()
            key_name = kwargs.get("log_name", None)
            if key_name: runtime_var[key_name] = runtime_var.get(key_name, 0) + end_time - start_time
            # print(f'Function {func.__name__} Took {end_time - start_time:.4f} seconds')
            return result
        return timeit_wrapper
    return _timer


def _merge_holders(curr_holder, total_holder):
    for name in curr_holder:
        if name not in total_holder:
            total_holder[name] = curr_holder[name]
        else:
            if isinstance(curr_holder[name], float):
                total_holder[name] = total_holder.get(name, 0) + curr_holder[name]
            else:
                total_holder[name] = torch.cat([total_holder[name],curr_holder[name]],axis=0)
    return total_holder



if __name__ == "__main__":
    matrix = torch.rand(768, 100)
    num_component = 99

    svd(matrix, num_component)

