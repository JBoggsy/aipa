from abc import ABC, abstractmethod
import json
import re


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model(ABC):
    def __init__(self, device: str | None = None):
        self.model = None
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.tools = {}

    @abstractmethod
    def generate(self, messages: list,
                 max_length: int = 2048,
                 reasoning: bool = False) -> str:
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def split_thinking(self, response: str) -> tuple[str, str]:
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def add_tool(self, tool_schema: dict, tool_function: callable):
        raise NotImplementedError("Subclasses must implement this method.")
    

class HFAutoModel(Model):
    TOKENIZERS = {}
    MODELS = {}

    def __init__(self, model_name: str):
        super().__init__()
        if model_name not in HFAutoModel.TOKENIZERS:
            HFAutoModel.TOKENIZERS[model_name] = AutoTokenizer.from_pretrained(model_name)
        if model_name not in HFAutoModel.MODELS:
            HFAutoModel.MODELS[model_name] = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = HFAutoModel.TOKENIZERS[model_name]
        self.model = HFAutoModel.MODELS[model_name].to(self.device)

    @property
    def tool_dicts(self) -> list:
        return [tool["tool_dict"] for tool in self.tools.values()]

    def add_tool(self, tool_schema: dict, tool_function: callable):
        tool_name = tool_schema["name"]
        self.tools[tool_name] = {
            "tool_dict": tool_schema,
            "function": tool_function
        }

    def parse_tool_calls(self, response: str) -> list:
        tool_call_pattern = re.compile(r"\<tool_call\>(.*?)\<\/tool_call\>", re.DOTALL)
        matches = tool_call_pattern.findall(response)
        tool_calls = []
        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue

        response = tool_call_pattern.sub("", response).strip()
        return tool_calls, response
    
    def execute_tool_call(self, tool_call: dict) -> str:
        tool_name = tool_call["name"]
        parameters = tool_call["arguments"]
        if tool_name in self.tools:
            tool_function = self.tools[tool_name]["function"]
            return tool_function(**parameters)
        else:
            raise ValueError(f"Tool '{tool_name}' not found.")

    def split_thinking(self, response: str) -> tuple[str, str]:
        thinking_pattern = re.compile(r"\<think\>(.*?)\<\/think\>", re.DOTALL)
        match = thinking_pattern.search(response)
        if match:
            thinking = match.group(1).strip()
            rest = thinking_pattern.sub("", response).strip()
            return thinking, rest
        return "", response
        
    def generate(self,
                 messages: list, 
                 max_length: int = 2048,
                 reasoning: bool = False) -> tuple[str, str]:
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            tools=list(self.tools.values()),
            enable_thinking=reasoning
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        output_tokens = self.model.generate(**model_inputs, 
                                            max_new_tokens=max_length)[0]
        response_tokens = output_tokens[len(model_inputs.input_ids[0]):]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        thinking, response = self.split_thinking(response)
        tool_calls, response = self.parse_tool_calls(response)
        if len(tool_calls) > 0:
            for tool_call in tool_calls:
                tool_response = self.execute_tool_call(tool_call)
                response += tool_response + "\n"
        return thinking, response.strip()