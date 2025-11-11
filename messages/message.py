from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolCall:
    """
    Represents a tool call from an LLM.
    
    Attributes:
        name (str): The name of the tool to call
        arguments (dict): The arguments to pass to the tool function
        id (str | None): Optional identifier for the tool call (used by some LLM providers)
    """
    name: str
    arguments: dict
    id: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "arguments": self.arguments
        }
        if self.id is not None:
            result["id"] = self.id
        return result


@dataclass
class Message:
    role: str
    content: str
    thinking: str = ""
    tool_calls: Optional[list['ToolCall']] = None

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "thinking": self.thinking,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls] if self.tool_calls else None
        }