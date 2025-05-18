import yaml
import os.path


if not os.path.isfile("config.yaml"):
    raise AssertionError("Config file was not found. Please ensure there is a config.yaml file in the current directory")
    
config_file = open("config.yaml")
configurations = yaml.safe_load(config_file)["configurations"]

model_configs = configurations["model_configs"]
model_name = model_configs["model_name"]
dataset_dir = model_configs["dataset_dir"]
batch_size = model_configs["batch_size"]
learning_rate = float(model_configs["learning_rate"])
split = model_configs["split"]
epochs = int(model_configs["epochs"])
checkpoints_dir = model_configs["checkpoints_dir"]

tokenizer_configs = configurations["tokenizer_configs"]
tokenizer_max_length = int(tokenizer_configs["max_length"])
tokenizer_truncate = bool(tokenizer_configs["truncate"])
tokenizer_padding = tokenizer_configs["padding"]
