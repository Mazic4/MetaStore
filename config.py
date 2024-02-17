import yaml


def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def merge_configs(cmd_args):

    yaml_config_path = cmd_args.config
    yaml_config = read_yaml_config(yaml_config_path)

    for key, value in vars(cmd_args).items():
        if key in ["target_model", "data"]:
            #exact config
            new_config = yaml_config[key].get(value, None)
            if new_config is not None:
                yaml_config[key] = new_config
            else:
                raise ValueError("Cannot find value {} with key {} in yaml.".format(value, key))

        if key == "num_analyzed_samples":
            yaml_config["data"][key] = value

        if key in ["num_query","method"]:
            yaml_config["system"][key] = value

    return yaml_config