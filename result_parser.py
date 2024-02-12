import collections
import os
import sys

import pickle

import numpy as np



def parse_result(target_dir):

    path_list = os.listdir(target_dir)
    result_dict = {}

    for path in path_list:
        if ".out" in path:
            continue
        else:
            file_list = os.listdir(os.path.join(target_dir, path))

            args = path.split("_")
            # print (args)
            experiment_type = args[2]
            method = args[4]
            dataset = args[6]
            num_samples = args[9]
            num_queries = args[12]
            if experiment_type not in result_dict:
                result_dict[experiment_type] = collections.defaultdict(dict)
            experiment_name = [dataset, num_samples, num_queries]
            if experiment_type == "layer":
                analyze_layer_type = args[4]
                hidden_size = args[5]
                experiment_name.append(analyze_layer_type)
                experiment_name.append(hidden_size)
            elif experiment_type == "batch":
                num_pseudo_samples = args[4]
                experiment_name.append(num_pseudo_samples)
            elif experiment_type == "model" and len(args) > 13:
                experiment_name.append(args[14])

            experiment_name = "_".join(experiment_name)

            for _file_name in file_list:
                file_name = os.path.join(target_dir, path, _file_name)
                if "total_meta_gradient" in file_name: continue
                # print (file_name)
                if not file_name.endswith(".pickle"): continue
                file = pickle.load(open(file_name, 'rb'))

                result_name, method = _file_name.split("_")[0], _file_name.split("_")[-1][:-7]
                if method not in result_dict[experiment_type][experiment_name]:
                    result_dict[experiment_type][experiment_name][method]= {}
                result_dict[experiment_type][experiment_name][method][result_name] = file

    for experiment_type in sorted(result_dict):
        # if experiment_type != "batch": continue
        print (experiment_type)
        for experiment_name in sorted(result_dict[experiment_type]):
            # if experiment_name.endswith("recon"): continue
            print ("\n")

            print (experiment_name)
            for method in result_dict[experiment_type][experiment_name]:
                if method.startswith("hparams"):
                    continue
                print (method)
                # print("\n")
                # print(result_dict[experiment_type][experiment_name][method])
                for key in result_dict[experiment_type][experiment_name][method]:
                    # if key != "calculation": continue
                    print (key)
                    print (result_dict[experiment_type][experiment_name][method][key])
                    if isinstance(result_dict[experiment_type][experiment_name][method][key], float):
                        print (result_dict[experiment_type][experiment_name][method][key])
                    else:
                        sum_time = 0
                        for layer in result_dict[experiment_type][experiment_name][method][key]:
                            sum_time += result_dict[experiment_type][experiment_name][method][key][layer]
                            print (layer, result_dict[experiment_type][experiment_name][method][key][layer])


if __name__ == "__main__":
    target_dir = "/home/zhanghuayi01/TED/ted_v3/experiment_result_test_model_single_query_10000"

    parse_result(target_dir)