from transformers import AutoTokenizer
import os
import random
from config import model_name, tokenizer_padding, tokenizer_max_length, tokenizer_truncate
import textwrap


class LanguageTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        special_tokens = [
            "<|begin_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>"
        ]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens})
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        

    def get_tokenizer(self):
        return self.tokenizer

    def save_pretrained(self, **kwargs):
        self.tokenizer.save_pretrained(**kwargs)

    def tokenize_for_inference(self, item):
        return self._tokenize(item=item)

    def tokenize_for_training(self, item):
        tokenized = self._tokenize(item=item)
        tokenized["input_ids"] = tokenized["input_ids"].squeeze()
        tokenized["attention_mask"] = tokenized["attention_mask"].squeeze()
        tokenized["labels"] = tokenized["labels"].squeeze()
        return tokenized
    
    def _tokenize(self, item):
        minified_file_content = item["minified_file_content"]
        original_file_content = item["original_file_content"]
        formatted_minified_content = self._format_llama(minified_file_content)
        return self.tokenizer(formatted_minified_content, text_target=original_file_content, padding=tokenizer_padding, 
                              max_length=tokenizer_max_length, truncation=tokenizer_truncate, return_tensors="pt")
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _get_random_system_prompt(self):
        system_prompts = [
            "You are an expert Javascript coder who is really good at deciphering minified Javascript code, but you only only in plain Javascript with no added text"
            "You are an expert in reading minified JavaScript, but you stick to pure JavaScript without adding explanations or extra context",
            "You really know your way around minified JavaScript, but you only write in plain JavaScript—no added notes, no extra chatter",
            "Act as a JavaScript expert who can expertly read minified code, but respond only using plain JavaScript, without any added context or commentary."
        ]
        return random.choice(system_prompts)

    def _format_llama(self, input):
        formatted_prompt = f"""
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            {self._get_random_system_prompt()}
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Deminify this code sample for me: 
            {input}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """

        return textwrap.dedent(formatted_prompt)


if __name__ == "__main__":
    tokenizer = LanguageTokenizer()
    json = {
        "hello_world": "some message {} \n why tho",
        "more_message": "some_more_message"
    }
    tokenized = tokenizer.tokenizer(
        json["hello_world"], text_target=json["more_message"], return_tensors="pt")
    print(tokenized)
