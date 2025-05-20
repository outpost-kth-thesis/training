from config import dataset_dir, batch_size
from torch.utils.data import Dataset, DataLoader
import os
from tokenization import LanguageTokenizer
import torch

class LanguageDataset(Dataset):
    all_files = []
    
    def __init__(self):
        self.t = LanguageTokenizer()
        self._walk_root_directory()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        minified_filepath = self.all_files[index]
        original_filepath = minified_filepath.replace("_terser.min.js", ".js") if "_terser" in minified_filepath else minified_filepath.replace("_google.min.js", ".js")
        minified_file_content = open(file=minified_filepath).read()
        original_file_content = open(file=original_filepath).read()
        json = {
            "minified_file_content":minified_file_content,
            "original_file_content":original_file_content
        }
        return self.t.tokenize_for_training(json)
        # return json

    def _walk_root_directory(self):
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if ".min.js" in file:
                    self.all_files.append(os.path.join(root, file))

def collate_fn(batches):
    # for batch in batches:
    #     batch["input_ids"] = batch["input_ids"].squeeze()
    #     batch["attention_mask"] = batch["attention_mask"].squeeze()
    #     batch["labels"] = batch["labels"].squeeze()

    # batches["input_ids"]
    return batches

if __name__ == "__main__":
    dt = LanguageDataset()
    torch.set_printoptions(threshold=float('inf'))
    dataloader = DataLoader(
        dataset=dt,
        batch_size=batch_size,
        shuffle=False,
        # collate_fn=collate_fn
    )

    for batch in dataloader:
        # batch["attention_mask"] = batch["attention_mask"].squeeze()
        # print(batch.keys())
        print(batch["input_ids"].size())
        # print(batch)
        break