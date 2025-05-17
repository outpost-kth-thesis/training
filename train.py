from data import LanguageDataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, get_scheduler, BitsAndBytesConfig, AutoModelForCausalLM
from config import epochs, model_name
from tokenization import LanguageTokenizer
from torch.optim import AdamW
import torch
from peft import LoraConfig, TaskType

file = open(".garbage/log.csv", "a")


class ModelTrainer(Trainer):
    def train(self):
        if self.optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
        if self.lr_scheduler is None:
            num_training_steps = len(self.train_dataset) * self.args.num_train_epochs
            self.lr_scheduler = get_scheduler(
                name="linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps,
            )

        for epoch in range(self.args.num_train_epochs):
            print(f"Starting epoch {epoch + 1}")

            for step, batch in enumerate(self.train_dataset):
                batch.to("cuda")
                if batch["input_ids"].size() != batch["labels"].size():
                    raise AssertionError("input_ids size does not match labels")
                
                assert torch.max(batch["input_ids"]) < self.model.config.vocab_size, "Found OOV token ID"
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                global file
                file.write(f"{step}, {loss.item()}\n")
                if step % self.args.logging_steps == 0:
                    print(f"Step {step}: Loss = {loss.item()}")

                
                if step % self.args.save_steps == 0:
                    torch.save(self.model.state_dict(), f"{self.args.output_dir}/model.pth")

            torch.save(self.model.state_dict(), f"{self.args.output_dir}/model.pth")



def train_loop():
    dataset = LanguageDataset()

    quantization_configs = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quantization_configs)
    
    training_args = TrainingArguments(
        output_dir="./training_output",
        per_device_train_batch_size=1,
        num_train_epochs=epochs,
        save_steps=1000,
        logging_steps=10
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # type of task to train on
        inference_mode=False, # set to False for training
        r=8, # dimension of the smaller matrices
        lora_alpha=32, # scaling factor
        lora_dropout=0.1 # dropout of LoRA layers
    )

    model.add_adapter(lora_config)

    t = LanguageTokenizer().tokenizer
    model.resize_token_embeddings(len(t))

    data_collator = DataCollatorForLanguageModeling(tokenizer=t, mlm=False)

    trainer = ModelTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()
    


if __name__ == "__main__":
    train_loop()