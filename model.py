import torch
from config import model_name, learning_rate
import lightning as L
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

class LanguageModel(L.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        quantization_configs = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_configs=quantization_configs)


    def forward(self, **kwargs):
        return self.model(kwargs)
    
    
    def training_step(self, **kwargs):
        output = self.model(kwargs)
        loss = output.loss
        self.log("training_loss", loss)
        return loss
    

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=learning_rate)
        