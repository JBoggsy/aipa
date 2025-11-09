from abc import ABC, abstractmethod

import torch


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
    def add_tool(self, tool_schema: dict, tool_function: callable):
        raise NotImplementedError("Subclasses must implement this method.")