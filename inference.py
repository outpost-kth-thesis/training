from transformers.trainer_utils import get_last_checkpoint
from config import checkpoints_dir, model_name
from transformers import AutoModelForCausalLM
from tokenization import LanguageTokenizer
import torch
from peft import PeftModel

last_checkpoint = get_last_checkpoint(checkpoints_dir)

t = LanguageTokenizer()
tokenizer = t.get_tokenizer()

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer), model.model.embed_tokens.num_embeddings)
model.to("cuda")


peft_model = PeftModel.from_pretrained(model, last_checkpoint)
peft_model.to("cuda")

test_function = """function findFirstAndLast(arr,target){let first=-1,last=-1;let left=0,right=arr.length-1;while(left<=right){let mid=Math.floor((left+right)/2);if(arr[mid]===target){first=mid;right=mid-1}else if(arr[mid]<target){left=mid+1}else{right=mid-1}}left=0;right=arr.length-1;while(left<=right){let mid=Math.floor((left+right)/2);if(arr[mid]===target){last=mid;left=mid+1}else if(arr[mid]<target){left=mid+1}else{right=mid-1}}return{first:first,last:last}}"""

json = {
    "minified_file_content":test_function,
    "original_file_content":""
}

tokens = t.tokenize_for_inference(json)
del tokens["labels"]
tokens.to("cuda")

with torch.no_grad():
    output_tokens = peft_model.generate(**tokens)
    decoded_output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(decoded_output)