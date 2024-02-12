"""
Store the Runtime log
"""

class RuntimeLog:
    def __init__(self):
        self.preprocess_time = {}
        self.io_time_per_layer = {}
        self.cal_time_per_layer = {}
        self.end2end_query_time = {}

    def save(self, log_base_dir):
        pass

logger = RuntimeLog()