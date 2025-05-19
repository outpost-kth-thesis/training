from data import LanguageDataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from config import epochs, model_name, batch_size
from tokenization import LanguageTokenizer
from torch.optim import AdamW
import torch
from peft import LoraConfig, TaskType, PeftModel, PeftConfig, get_peft_model
from safetensors.torch import save_model
import bitsandbytes as bnb
from peft.optimizers import create_loraplus_optimizer

class ModifiedTrainer(Trainer):
    # def __init__(self, model = None, args = None, data_collator = None, train_dataset = None, eval_dataset = None, processing_class = None, model_init = None, compute_loss_func = None, compute_metrics = None, callbacks = None, optimizers = ..., optimizer_cls_and_kwargs = None, preprocess_logits_for_metrics = None):
    #     # super().__init__(model, args, data_collator, train_dataset, eval_dataset, processing_class, model_init, compute_loss_func, compute_metrics, callbacks, optimizers, optimizer_cls_and_kwargs, preprocess_logits_for_metrics)
        
    # #     if not hasattr(self.model, "peft_config"):
    # #         print("MODEL MODEL MODEL does not have peft config")
    # #     else:
    # #         print("MODEL MODEL MODEL PEFT adapter already attached.")
    #     print("something is it working")
    
    def _save_optimizer_and_scheduler(self, output_dir):
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'param_groups': self.optimizer.param_groups,
            'state': self.optimizer.state
        }, f"{output_dir}/optimizer.pt")


    def save_model(self, output_dir = None, _internal_call = False):
        self.processing_class.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)

    def train(self, resume_from_checkpoint = None, trial = None, ignore_keys_for_eval = None, **kwargs):
        print("checkpoint", resume_from_checkpoint)
        super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        print("loading from checkpoint called")
        optimizer_state_dict = torch.load(
            f"{resume_from_checkpoint}/optimizer.pt",
            weights_only=False                          # Setting this line to False is generally not recommended as this can allow for arbitrary code execution
        )


        # for each in self.optimizer.param_groups:
        #     for i in each["params"]:
        #         print(i)
        #         print(i.data.size())
        # print("done printing")

        
        # for each in optimizer_state_dict["param_groups"]:
        #     for i in each["params"]:
        #         print(i.data.size())
        # print("ok printing")

        if not hasattr(self.model.base_model, "peft_config"):
            print("MODEL MODEL MODEL does not have peft config")
        else:
            print("MODEL MODEL MODEL PEFT adapter already attached.")
        self.optimizer.load_state_dict(optimizer_state_dict)
        # self.peft_model = PeftModel.from_pretrained(self.model.base_model, resume_from_checkpoint)
        # self.model = self.peft_model
        # self.processing_class = AutoTokenizer.from_pretrained(resume_from_checkpoint)
        
dataset = LanguageDataset()

quantization_configs = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quantization_configs)
t = LanguageTokenizer().tokenizer
model.resize_token_embeddings(len(t), model.model.embed_tokens.num_embeddings)



lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

peft_model = get_peft_model(model=model, peft_config=lora_config)

print("is there where the log is coming from?")

optimizer = create_loraplus_optimizer(
    model=peft_model,
    optimizer_cls=bnb.optim.Adam8bit,
    lr=5e-5,
    loraplus_lr_ratio=16,
)
# print("is there where the log is coming from? second before")

# loaded_peft_model = PeftModel.from_pretrained(model, "./training_output/checkpoint-20")

# print("is there where the log is coming from? second")

training_args = TrainingArguments(
    output_dir="./training_output",
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    save_steps=10,
    save_strategy="steps",
    save_safetensors=False,
    logging_steps=10,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=t, mlm=False)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000,
)
t = AutoTokenizer.from_pretrained("./training_output/checkpoint-20")
trainer = ModifiedTrainer(
    model=peft_model,
    processing_class=t,
    args=training_args,
    optimizers=(optimizer, scheduler),
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train(resume_from_checkpoint=True)