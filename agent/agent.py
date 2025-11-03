import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)