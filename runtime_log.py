"""
Store the Runtime log
"""
import pickle
import yaml
import os

from datetime import datetime

class RuntimeLog:
    def __init__(self):
        self.preprocess_time = {}
        self.io_time_per_layer = {}
        self.cal_time_per_layer = {}
        self.end2end_query_time = {}

    def save(self, _log_base_dir, config, meta_gradient):

        today_date = datetime.now().date()
        method = config["system"]["method"]
        os.makedirs(_log_base_dir, exist_ok=True)

        log_base_dir_suffix = "{}/{}_{}_{}_{}".format(str(today_date),
                                                      config["system"]["method"],
                                                      config["data"]["data_name"],
                                                      config["data"]["num_analyzed_samples"],
                                                      config["system"]["num_query"])
        log_base_dir = os.path.join(_log_base_dir,log_base_dir_suffix)
        os.makedirs(log_base_dir, exist_ok=True)

        file_name = os.path.join(log_base_dir, "config.yaml")
        print (file_name)
        with open(file_name, "w") as file:
            yaml.dump(config, file, allow_unicode=True)

        file_name = os.path.join(log_base_dir, "total_meta_gradient_{}.pickle".format(config["system"]["method"]))
        print(file_name)
        with open(file_name, 'wb') as handle:
            pickle.dump(meta_gradient, handle)

        file_name = os.path.join(log_base_dir, "e2e_query_timer_{}.pickle".format(method))
        print(file_name)
        with open(file_name, 'wb') as handle:
            pickle.dump(self.end2end_query_time, handle)

        file_name = os.path.join(log_base_dir, "calculation_query_timer_{}.pickle".format(method))
        print(file_name)
        with open(file_name, 'wb') as handle:
            pickle.dump(self.cal_time_per_layer, handle)

        file_name = os.path.join(log_base_dir, "io_query_timer_{}.pickle".format(method))
        print(file_name)
        with open(file_name, 'wb') as handle:
            pickle.dump(self.io_time_per_layer, handle)

        file_name = os.path.join(log_base_dir, "preprocess_timer_{}.pickle".format(method))
        print(file_name)
        with open(file_name, 'wb') as handle:
            pickle.dump(self.preprocess_time, handle)

    def print(self):
        print (self.preprocess_time)
        print (self.io_time_per_layer)
        print (self.cal_time_per_layer)
        print (self.end2end_query_time)


logger = RuntimeLog()