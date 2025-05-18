from data import LanguageDataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from config import epochs, model_name, batch_size
from tokenization import LanguageTokenizer
from torch.optim import AdamW
import torch
from peft import LoraConfig, TaskType, PeftModel, PeftConfig, get_peft_model
from safetensors.torch import save_model


def train_loop():

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

    training_args = TrainingArguments(
        output_dir="./training_output",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_steps=10,
        save_strategy="steps",
        save_safetensors=False,
        logging_steps=10,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False, 
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    peft_model = get_peft_model(model=model, peft_config=lora_config)


    data_collator = DataCollatorForLanguageModeling(tokenizer=t, mlm=False)
    trainer = Trainer(
        model=peft_model,
        processing_class=t,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )


    trainer.train()


if __name__ == "__main__":
    train_loop()