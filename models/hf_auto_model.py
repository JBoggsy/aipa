import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.model import Model
from messages import Message


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
                 temperature: float = 0.7,
                 reasoning: bool = False) -> Message:
        """
        Generates a response from the model based on the provided messages.

        Args:
            messages (list): A list of message dictionaries containing 'role' and 'content'.
            max_length (int, optional): The maximum length of the generated response. Defaults to 2048.
            temperature (float, optional): The sampling temperature for generation. Defaults to 0.8.
            reasoning (bool, optional): Whether to enable reasoning capabilities. Defaults to
            False

        Returns:
            Message: A Message object containing the response, thinking process, and tool calls.
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            tools=list(self.tools.values()),
            enable_thinking=reasoning
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        output_tokens = self.model.generate(**model_inputs, 
                                            max_new_tokens=max_length,
                                            temperature=temperature,
                                            top_p=0.95)[0]
        response_tokens = output_tokens[len(model_inputs.input_ids[0]):]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        thinking, response = self.split_thinking(response)
        tool_calls, response = self.parse_tool_calls(response)
        return Message(
            role="assistant",
            content=response.strip(),
            thinking=thinking,
            tool_calls=tool_calls if tool_calls else None
        )