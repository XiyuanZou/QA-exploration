import json
from typing import List, Dict, Optional
import torch
import pandas
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import time

class BaseGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_answers(self, input_texts: List[str]) -> List[List[str]]: #return (batch_size, num_sampling, str_length)
        raise NotImplementedError


class HFGenerator(BaseGenerator):
    def __init__(self, model_name: str, device: Optional[str] = None):
        super().__init__(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, load_in_8bit=True, cache_dir='/network/scratch/x/xiyuan.zou/cache/transformers_cache')
        
        
        #if torch.cuda.is_available():
            #self.model = self.model.cuda()

    def generate_answers(
        self,
        input_texts: List[str],
        max_input_length: int = 100000,
        max_new_tokens: int = 100,
        num_sampling: int = 1,
        greedy_decoding: bool = False,
    ) -> List[str]:
        
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            padding_side='left',
            truncation=True,
            max_length=max_input_length,
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if greedy_decoding:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        else:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_return_sequences=num_sampling,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        outputs = outputs.reshape(len(input_texts), num_sampling, -1) #(batch_size, num_sampling, -1)  
        answers = []
        for i, input_text in enumerate(input_texts):
            multi_samplings=[]
            for j in range(num_sampling):
                answer_text = self.tokenizer.decode(outputs[i,j], skip_special_tokens=True)
                answer_text = answer_text[len(input_text):].strip()
                multi_samplings.append(answer_text)
            answers.append(multi_samplings)
        return answers


class OpenAIGenerator(BaseGenerator):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        if "gpt" in model_name.lower():
            self.client = OpenAI(api_key=api_key)
        elif "deepseek" in model_name.lower():
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    def generate_answers( #generate answer for a single prompt
            self,
            input_texts: List[str],
            temperature: float = 0.9,
            max_new_tokens: int = 100,
            num_sampling: int = 1,
            OPENAI_MAX_WORKERS: int = 10, 
        ) -> List[str]:
        def _generate_single(input_text):
            for attempt in range(500):  # Loop: max 500 tries
                try:
                    if self.model_name == "o3-mini":
                        responses = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": input_text}
                            ],
                            n=num_sampling,
                            max_completion_tokens=max_new_tokens,
                        )
                    else:
                        responses = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": input_text}
                            ],
                            n=num_sampling,
                            temperature=temperature,
                            max_tokens=max_new_tokens,
                        )
                    return [choice.message.content for choice in responses.choices]
                except openai.RateLimitError:
                    print("Rate limit hit. Retrying...")
                    time.sleep(30)
        
        with ThreadPoolExecutor(max_workers=OPENAI_MAX_WORKERS) as executor:
            answers = list(executor.map(_generate_single, input_texts))
        return answers
