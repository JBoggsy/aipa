from abc import ABC, abstractmethod

import torch

from messages import Message, ToolCall


class Model(ABC):
    def __init__(self, device: str | None = None):
        self.model = None
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.tools = {}

    @abstractmethod
    def generate(self, messages: list[Message],
                 max_length: int = 2048,
                 temperature: float = 0.8,
                 reasoning: bool = False,
                 format: str | None = None) -> Message:
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def parse_tool_calls(self, raw_tool_calls) -> list[ToolCall] | None:
        """
        Parse tool calls from the LLM's raw response format into ToolCall instances.
        
        Args:
            raw_tool_calls: The raw tool call data from the LLM (format varies by provider)
            
        Returns:
            list[ToolCall] | None: A list of ToolCall instances, or None if no tool calls
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def add_tool(self, tool_schema: dict, tool_function: callable):
        raise NotImplementedError("Subclasses must implement this method.")