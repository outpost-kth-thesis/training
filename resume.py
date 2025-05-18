from data import LanguageDataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from config import epochs, model_name, batch_size
from tokenization import LanguageTokenizer
from torch.optim import AdamW
import torch
from peft import LoraConfig, TaskType, PeftModel, PeftConfig
from safetensors.torch import save_model


class ModifiedTrainer(Trainer):
    def _save_optimizer_and_scheduler(self, output_dir):
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'param_groups': self.optimizer.param_groups,
            'state': self.optimizer.state
        }, f"{output_dir}/optimizer.pt")



dataset = LanguageDataset()

quantization_configs = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quantization_configs)
t = LanguageTokenizer().tokenizer
model.resize_token_embeddings(len(t), mean_resizing=True)

peft_model = PeftModel.from_pretrained(model, "./training_output/checkpoint-20")

training_args = TrainingArguments(
    output_dir="./training_output",
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    save_steps=10,
    save_strategy="steps",
    save_safetensors=False,
    logging_steps=10,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=t, mlm=False)

t = AutoTokenizer.from_pretrained("./training_output/checkpoint-20")
trainer = Trainer(
    model=peft_model,
    processing_class=t,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train(resume_from_checkpoint=True)