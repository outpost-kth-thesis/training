# # model_name="Qwen/Qwen3-1.7B"
# model_name="meta-llama/Llama-3.1-8B"
# dataset_dir="./.garbage/downloadable"
# batch_size=1
# learning_rate=5e-5
# split="train"
# tokenizer_padding_policy="max_length"
# tokenizer_max_length=512
# truncate_tokenizer_output=True
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

tokenizer_configs = configurations["tokenizer_configs"]
tokenizer_max_length = int(tokenizer_configs["max_length"])
tokenizer_truncate = bool(tokenizer_configs["truncate"])
tokenizer_padding = tokenizer_configs["padding"]
