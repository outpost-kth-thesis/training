from config import dataset_dir
from torch.utils.data import Dataset, DataLoader
import os
from datasets import Dataset
from tokenization import LanguageTokenizer
import torch

class LanguageDataset(Dataset):
    def __init__(self):
        self.all_content = []
        self.all_files = []
        self.t = LanguageTokenizer()
        self._walk_root_directory()

    def __getitem__(self, index):
        return self.all_content[index]

    def _walk_root_directory(self):
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if ".min.js" in file:
                    original_filepath = file.replace("_terser.min.js", ".js") if "_terser" in file else file.replace("_google.min.js", ".js")
                    minified_file_content = open(file=os.path.join(root, file)).read()
                    original_file_content = open(file=os.path.join(root, original_filepath)).read()
                    json = {
                        "minified_file_content":minified_file_content,
                        "original_file_content":original_file_content
                    }
                    tokenized = self.t.tokenize(json)
                    self.all_content.append(tokenized)

if __name__ == "__main__":
    dt = LanguageDataset()
    torch.set_printoptions(threshold=float('inf'))
    print(dt.__getitem__(2)["input_ids"])