from config import dataset_dir
from torch.utils.data import Dataset
import os
from tokenization import LanguageTokenizer
import torch

class LanguageDataset(Dataset):
    all_files = []
    t = LanguageTokenizer()
    
    def __init__(self):
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
        return self.t.tokenize(json)

    def _walk_root_directory(self):
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if ".min.js" in file:
                    self.all_files.append(os.path.join(root, file))

if __name__ == "__main__":
    dt = LanguageDataset()
    torch.set_printoptions(threshold=float('inf'))
    for i, each in enumerate(dt):
        print(each)
        break