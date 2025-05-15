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

        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quantization_configs)
        self.model.to("cuda")


    def forward(self, **kwargs):
        return self.model(kwargs)
    

    # def training_step(self, input_ids, attention_mask, labels):
    #     output = self.model(input_ids, attention_mask, labels)
    #     loss = output.loss
    #     self.log("training_loss", loss)
    #     return loss
    

    # def configure_optimizers(self):
    #     return torch.optim.AdamW(self.parameters(), lr=learning_rate)
        


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