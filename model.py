import torch
from config import model_name
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from data import LanguageDataset
from tokenization import LanguageTokenizer
import torch.nn as nn

class LanguageModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        quantization_configs = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", quantization_config=quantization_configs)


    def forward(self, input_ids, attention_mask, labels):
        if input_ids == None or labels == None:
            raise AssertionError("either input_ids or labels have not been passed")
        
        if attention_mask == None:
            raise AssertionError("Attention mask has not been passed")
        
        return self.model(input_ids, attention_mask, labels)



if __name__ == "__main__":
    dataset = LanguageDataset()
    t = LanguageTokenizer()
    item = dataset.__getitem__(0)
    item.to("cuda")
    print("loaded dataset")
    model = LanguageModel()
    output = model(**item)
    print((output))
    print(output.loss)