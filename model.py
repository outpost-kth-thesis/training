import torch
from config import model_name
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from data import LanguageDataset
import torch.nn as nn
from tokenization import LanguageTokenizer

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


    def forward(self, item):
        
        # if input_ids == None or labels == None:
        #     raise AssertionError("either input_ids or labels have not been passed")
        
        # if attention_mask == None:
        #     raise AssertionError("Attention mask has not been passed")
        

        # t = LanguageTokenizer()
        # item = t.tokenize(json)
        # item.to("cuda")
        return self.model(**item)



if __name__ == "__main__":
    dataset = LanguageDataset()
    item = dataset.__getitem__(0)
    
    # json = {
    #     "minified_file_content":"console.log(\"hello world\")",
    #     "original_file_content":"some more bullshit idk"
    # }
    # t = LanguageTokenizer()
    # tok = t.get_tokenizer()
    # item = tok("hello world", return_tensors="pt")
    # item["labels"] = item["input_ids"]
    # print(item)
    # item.to("cuda")
    
    # print("loaded dataset")
    model = LanguageModel()

    quantization_configs = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quantization_configs)
    model.to("cuda")
    # item = LanguageTokenizer().tk("hello world")
    item.to("cuda")

    # model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    output = model(**item)
    print((output))
    print(output.loss)



# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
# tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

# inputs = tokenizer("hello world", return_tensors="pt")
# inputs["labels"] = inputs["input_ids"]

# outputs = model(**inputs)
# print(outputs.loss, outputs.logits.shape)