from model import LanguageModel
from data import LanguageDataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, get_scheduler
from config import epochs
from tokenization import LanguageTokenizer
from torch.optim import AdamW


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
                outputs = self.model(**batch)
                print(outputs)
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if step % self.args.logging_steps == 0:
                    print(f"Step {step}: Loss = {loss.item()}")



def train_loop():
    dataset = LanguageDataset()
    dataloader = DataLoader(dataset)
    model = LanguageModel()
    t = LanguageTokenizer()

    training_args = TrainingArguments(
        output_dir="./training_output",
        per_device_train_batch_size=1,
        num_train_epochs=epochs,
        save_steps=100,
        logging_steps=10
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=t.tokenizer, mlm=False)

    trainer = ModelTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()
    


if __name__ == "__main__":
    train_loop()