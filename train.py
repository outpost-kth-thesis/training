from data import LanguageDataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from config import epochs, model_name, batch_size, checkpoints_dir
from tokenization import LanguageTokenizer
from torch.optim import AdamW
import torch
from peft.optimizers import create_loraplus_optimizer
from peft import LoraConfig, TaskType, get_peft_model
import bitsandbytes as bnb

dataset = LanguageDataset()
t = LanguageTokenizer().tokenizer

quantization_configs = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", quantization_config=quantization_configs)
model.resize_token_embeddings(len(t), model.model.embed_tokens.num_embeddings)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

peft_model = get_peft_model(model=model, peft_config=lora_config)

optimizer = create_loraplus_optimizer(
    model=peft_model,
    optimizer_cls=bnb.optim.Adam8bit,
    lr=5e-5,
    loraplus_lr_ratio=16,
)

for each in optimizer.param_groups:
    for i in each["params"]:        
        print(i.data.size())
print("done printing before call")

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000,
)


training_args = TrainingArguments(
    output_dir=checkpoints_dir,
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    save_steps=10,
    save_strategy="steps",
    save_safetensors=False,
    logging_steps=1,
    lr_scheduler_type="linear"
)


data_collator = DataCollatorForLanguageModeling(tokenizer=t, mlm=False)
trainer = Trainer(
    model=peft_model,
    processing_class=t,
    optimizers=(optimizer, scheduler),
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)


trainer.train(resume_from_checkpoint=True)
