from abc import ABC, abstractmethod

import torch

from messages import Message


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
                 temperature: float = 0.8,
                 reasoning: bool = False,
                 format: str | None = None) -> Message:
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def add_tool(self, tool_schema: dict, tool_function: callable):
        raise NotImplementedError("Subclasses must implement this method.")