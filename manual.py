from data import LanguageDataset
from torch.utils.data import DataLoader
from transformers import Trainer, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, get_linear_schedule_with_warmup
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
from torch.utils.data import DataLoader
import os
from os import path

class CustomTrainer(Trainer):
    def __init__(self):
        self.dataset = LanguageDataset()
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size
        )
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


        num_training_steps = len(self.dataloader) * epochs
        num_warmup_steps = int(0.1 * num_training_steps)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train(self):
        os.makedirs(checkpoints_dir, exist_ok=True)
        last_checkpoint = get_last_checkpoint(checkpoints_dir)
        if last_checkpoint != None:
            self._resume_training()
        else:
            self._do_training()

    def _do_training(self):
        """
        Runs backwards propagation for each step for each epoch.
        """
        for each_epoch in range(epochs):
            for step, batch in enumerate(self.dataloader):
                batch.to("cuda")
                output = self.peft_model(**batch)
                loss = output.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                print(step, output.loss)

    def _resume_training(self):
        if path.exists(checkpoints_dir):
            last_checkpoint = get_last_checkpoint(checkpoints_dir)

        if last_checkpoint == None:
            return
        
        self._load_checkpoint()
        

    def _save_checkpoint(self, epochs):
        os.makedirs(checkpoints_dir, exist_ok=True)
        last_checkpoint = get_last_checkpoint(checkpoints_dir)
        if last_checkpoint == None:
            checkpoint_save = "checkpoint-1"
        else:
            checkpoint_save = "checkpoint-" + str(int(last_checkpoint.split("-")[1]) + 1)

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


        # for each in self.optimizer.param_groups:
        #     for i in each["params"]:
        #         print(i.data.size())
        #     print("done printing")

        
        # for each in optimizer_state_dict["param_groups"]:
        #     for i in each["params"]:
        #         print(i.data.size())


        self.optimizer.load_state_dict(optimizer_state_dict)

        if not hasattr(self.base_model, "peft_config"):
            model = PeftModel.from_pretrained(self.base_model, checkpoints_dir)
            print("does not have peft config")
        else:
            print("PEFT adapter already attached.")
        # self.peft_model = PeftModel.from_pretrained(self.base_model, f"{checkpoints_dir}")
        self.processing_class = AutoTokenizer.from_pretrained(f"{checkpoints_dir}")
        

trainer = CustomTrainer()
trainer._load_checkpoint()

# loaded = torch.load(f"{checkpoints_dir}/optimizer.pt")
# print(loaded.keys())