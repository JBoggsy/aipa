from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class Message:
    role: str
    content: str
    thinking: str = ""
    tool_calls: Optional[Union[dict, list]] = None

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "thinking": self.thinking,
            "tool_calls": self.tool_calls
        }