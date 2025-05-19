from trl import PPOConfig, PPOTrainer, create_reference_model
from transformers import AutoModelForCausalLM
from tokenization import LanguageTokenizer, BitsAndBytesConfig
from data import LanguageDataset
from config import model_name, batch_size
import torch

dataset = LanguageDataset()
t = LanguageTokenizer().tokenizer

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model.resize_token_embeddings(len(t), model.model.embed_tokens.num_embeddings)

reference_model = create_reference_model(model=model)



ppo_config = PPOConfig(
    batch_size=batch_size,
)

ppo_trainer = PPOTrainer(
    ppo_config=ppo_config,
    bnb_config=bnb_config,
)
