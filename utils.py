import json
import re
from typing import List, Dict, Optional
import pandas, spacy

def load_prompt(file_path):
    with open(file_path, "r") as f:
        return f.read().strip()
    
def load_results(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

def save_results(results: List[Dict], file_path: str):
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)

def split_text_into_sentences(text):
    nlp = spacy.load('en_core_web_sm')
    sentences = [str(i).strip() for i in nlp(text).sents]
    return sentences

def keep_first_sentence(text): 
    return split_text_into_sentences(text)[0]

def get_question(text):
    return text.split('?', 1)[0]