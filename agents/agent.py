import json
import re
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from agents.prompt_set import PromptSet, Prompt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Agent:
    def __init__(self, model_name: str, prompt_dir: str = "agents/prompts"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        self.prompt_set = PromptSet(prompt_dir)
        self.system_prompt = self.prompt_set["system_prompt"]()
        self.secrets = self.load_secrets()
        
    def load_secrets(self) -> dict:
        with open("config/secrets.json", "r") as file:
            return json.load(file)

    def make_simple_messages(self, user_prompt: str) -> list:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def split_thinking(self, response: str) -> tuple[str, str]:
        thinking_pattern = re.compile(r"\<think\>(.*?)\<\/think\>", re.DOTALL)
        match = thinking_pattern.search(response)
        if match:
            thinking = match.group(1).strip()
            rest = thinking_pattern.sub("", response).strip()
            return thinking, rest
        return "", response

    def generate_response(self, messages: list, max_length: int = 2048) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)
        output_tokens = self.model.generate(**model_inputs, max_new_tokens=32768)[0]
        response_tokens = output_tokens[len(model_inputs.input_ids[0]):]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)