from data import LanguageDataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from config import epochs, model_name, batch_size, learning_rate, checkpoints_dir
from tokenization import LanguageTokenizer
from torch.optim import AdamW
import torch
from peft.optimizers import create_loraplus_optimizer
from peft import LoraConfig, TaskType, PeftModel, PeftConfig, get_peft_model
from safetensors.torch import save_model
from transformers.trainer_utils import get_last_checkpoint
import bitsandbytes as bnb
from torch.optim import AdamW
import os
from os import path

class CustomTrainer(Trainer):
    def __init__(self, **kwargs):

        self.dataset = LanguageDataset()
        self.processing_class = LanguageTokenizer().tokenizer

        quantization_configs = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", quantization_config=quantization_configs)
        self.base_model.resize_token_embeddings(len(self.processing_class))
        
        lora_config = LoraConfig(
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.peft_model = get_peft_model(model=self.base_model, peft_config=lora_config)

        self.optimizer = create_loraplus_optimizer(
            model=self.base_model,
            optimizer_cls=AdamW,
            lr=learning_rate,
            loraplus_lr_ratio=16
        )
        self.scheduler = None

        

    def train(self):
        if path.exists(checkpoints_dir):
            last_checkpoint = get_last_checkpoint(checkpoints_dir)

        if last_checkpoint != None:
            self._resume_training()
        else:
            pass


    def _resume_training(self):
        if path.exists(checkpoints_dir):
            last_checkpoint = get_last_checkpoint(checkpoints_dir)

        if last_checkpoint == None:
            return
        

    def _save_checkpoint(self, epochs):
        os.makedirs("./checkpoints", exist_ok=True)
        last_checkpoint = get_last_checkpoint(checkpoints_dir)
        if last_checkpoint == None:
            checkpoint_save = "checkpoint-1"
        else:
            checkpoint_save = f"checkpoint-{int(last_checkpoint.split("-")[1]) + 1}"

        os.makedirs(f"{checkpoints_dir}/{checkpoint_save}", exist_ok=True)

        self.processing_class.save_pretrained(f"{checkpoints_dir}/")
        self.peft_model.save_pretrained(checkpoints_dir)
        torch.save({
            'epochs': epochs,
            'model': self.base_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'param_groups': self.optimizer.param_groups,
            'state': self.optimizer.state
        }, f"{checkpoints_dir}/{checkpoint_save}/model.pt")

    def _load_checkpoint(self):
        if path.exists(checkpoints_dir):
            last_checkpoint = get_last_checkpoint(checkpoints_dir)

        if last_checkpoint == None:
            print("No checkpoints found in the checkpoints directory")
            return
        
        optimizer_state_dict = torch.load(
            f"{last_checkpoint}/model.pt",
            weights_only=False                          # Setting this line to False is generally not recommended as this can allow for arbitrary code execution
        )
        self.optimizer.load_state_dict(optimizer_state_dict, save_embedding_layer=True)
        self.peft_model = PeftModel.from_pretrained(self.base_model, f"{checkpoints_dir}", save_embedding_layer=True)
        self.processing_class = AutoTokenizer.from_pretrained(f"{checkpoints_dir}", save_embedding_layer=True)
        




# dataset = LanguageDataset()
# t = LanguageTokenizer().tokenizer

# quantization_configs = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_name, device_map="auto", quantization_config=quantization_configs)
# model.resize_token_embeddings(len(t), model.model.embed_tokens.num_embeddings)

# training_args = TrainingArguments(
#     output_dir="./training_output",
#     per_device_train_batch_size=batch_size,
#     num_train_epochs=epochs,
#     save_steps=10,
#     save_strategy="steps",
#     save_safetensors=False,
#     logging_steps=10,
#     lr_scheduler_type="linear"
# )

# lora_config = LoraConfig(
#     # task_type=TaskType.CAUSAL_LM,
#     inference_mode=False,
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1,
#     target_modules=[
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "o_proj",
#         "gate_proj",
#         "up_proj",
#         "down_proj",
#         "lm_head",
#     ],
#     # trainable_token_indices={'embed_tokens': t.convert_tokens_to_ids('[PAD]')},
# )
# # lora_config = LoraConfig(
# #     target_modules='all-linear',
# #     trainable_token_indices={'embed_tokens': t.convert_tokens_to_ids('[PAD]')},
# # )

# peft_model = get_peft_model(model=model, peft_config=lora_config)

# optimizer = create_loraplus_optimizer(
#     model=model,
#     optimizer_cls=bnb.optim.Adam8bit,
#     lr=5e-5,
#     loraplus_lr_ratio=16,
# )

# scheduler = get_cosine_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=100,
#     num_training_steps=1000,
# )

# # data_collator = DataCollatorForLanguageModeling(tokenizer=t, mlm=False)
# trainer = Trainer(
#     model=peft_model,
#     processing_class=t,
#     optimizers=(optimizer, scheduler),
#     args=training_args,
#     train_dataset=dataset,
#     # data_collator=data_collator
# )


trainer = CustomTrainer()
trainer._save_checkpoint(1)
# trainer._load_checkpoint()

# loaded = torch.load(f"{checkpoints_dir}/optimizer.pt")
# print(loaded.keys())