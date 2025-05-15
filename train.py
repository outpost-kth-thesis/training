from model import LanguageModel
from data import LanguageDataset
from torch.utils.data import DataLoader
import lightning as L

def main():
    dataset = LanguageDataset()                                         # This will take a while to load depending on the size of the dataset
    model = LanguageModel()
    dataloader = DataLoader(dataset=dataset)
    trainer = L.Trainer()
    trainer.fit(model=model, train_dataloaders=dataloader)
    


if __name__ == "__main__":
    main()