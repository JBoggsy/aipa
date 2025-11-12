import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.model import Model
from messages import Message, ToolCall


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

    def parse_tool_calls(self, raw_tool_calls) -> list[ToolCall] | None:
        """
        Parse tool calls from HuggingFace's raw response format into ToolCall instances.
        
        For HuggingFace models, raw_tool_calls is expected to be a list of dicts parsed
        from the model's text output.
        
        Args:
            raw_tool_calls: List of tool call dictionaries or None
            
        Returns:
            list[ToolCall] | None: A list of ToolCall instances, or None if no tool calls
        """
        if not raw_tool_calls:
            return None
            
        tool_calls = []
        for tool_call_dict in raw_tool_calls:
            try:
                # Convert dict to ToolCall instance
                tool_calls.append(ToolCall(
                    name=tool_call_dict.get('name', ''),
                    arguments=tool_call_dict.get('arguments', {}),
                    id=tool_call_dict.get('id')
                ))
            except (AttributeError, TypeError):
                continue
        
        return tool_calls if tool_calls else None
    
    def extract_tool_calls_from_text(self, response: str) -> tuple[list[dict], str]:
        """
        Extract tool calls from the response text and return them as dicts.
        
        Args:
            response (str): The model's response text
            
        Returns:
            tuple[list[dict], str]: A tuple of (list of tool call dicts, cleaned response text)
        """
        tool_call_pattern = re.compile(r"\<tool_call\>(.*?)\<\/tool_call\>", re.DOTALL)
        matches = tool_call_pattern.findall(response)
        tool_call_dicts = []
        for match in matches:
            try:
                tool_call_dict = json.loads(match.strip())
                tool_call_dicts.append(tool_call_dict)
            except json.JSONDecodeError:
                continue

        response = tool_call_pattern.sub("", response).strip()
        return tool_call_dicts, response
    
    def split_thinking(self, response: str) -> tuple[str, str]:
        thinking_pattern = re.compile(r"\<think\>(.*?)\<\/think\>", re.DOTALL)
        match = thinking_pattern.search(response)
        if match:
            thinking = match.group(1).strip()
            rest = thinking_pattern.sub("", response).strip()
            return thinking, rest
        return "", response
        
    def generate(self,
                 messages: list[Message], 
                 max_length: int = 2048,
                 temperature: float = 0.7,
                 reasoning: bool = False,
                 format: str | None = None) -> Message:
        """
        Generates a response from the model based on the provided messages.

        Args:
            messages (list[Message]): A list of Message objects containing role and content.
            max_length (int, optional): The maximum length of the generated response. Defaults to 2048.
            temperature (float, optional): The sampling temperature for generation. Defaults to 0.8.
            reasoning (bool, optional): Whether to enable reasoning capabilities. Defaults to
            False
            format (str | None, optional): The output format for the response. Currently supports
            "json" for JSON-formatted output. Defaults to None.

        Returns:
            Message: A Message object containing the response, thinking process, and tool calls.
        """
        # Convert Message objects to dictionaries for the tokenizer
        message_dicts = [msg.to_dict() for msg in messages]
        
        # Add JSON formatting instruction if requested
        if format == "json":
            # Add JSON formatting instruction to the last user message
            modified_messages = message_dicts.copy()
            if modified_messages and modified_messages[-1]["role"] == "user":
                modified_messages[-1] = modified_messages[-1].copy()
                modified_messages[-1]["content"] += "\n\nPlease respond with valid JSON only."
            message_dicts = modified_messages
        
        text = self.tokenizer.apply_chat_template(
            message_dicts,
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
        tool_call_dicts, response = self.extract_tool_calls_from_text(response)
        tool_calls = self.parse_tool_calls(tool_call_dicts)
        return Message(
            role="assistant",
            content=response.strip(),
            thinking=thinking,
            tool_calls=tool_calls
        )