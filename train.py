from model import LanguageModel
from data import LanguageDataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from config import epochs
from tokenization import LanguageTokenizer


class ModelTrainer(Trainer):
    def train(self):
        pass

def train_loop():
    dataset = LanguageDataset()
    dataloader = DataLoader(dataset)
    model = LanguageModel()
    t = LanguageTokenizer()

    training_args = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=epochs,
        save_steps=100
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=t.tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()
    


if __name__ == "__main__":
    train_loop()