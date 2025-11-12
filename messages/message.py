from dataclasses import dataclass
from typing import Optional, Any


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
    result: Any = None
    id: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "arguments": self.arguments
        }
        if self.id is not None:
            result["id"] = self.id
        return result
    
    def to_message(self) -> 'Message':
        return Message(
            role="tool",
            content=f"{self.name}({self.arguments}):\n{self.result}"
        )

    def __str__(self):
        return f"ToolCall(name={self.name}, arguments={self.arguments}, result={self.result}, id={self.id})"

@dataclass
class Message:
    role: str
    content: str
    thinking: str = ""
    tool_calls: Optional[list['ToolCall']] = None

    def to_dict(self) -> dict:
        """
        Convert Message to dictionary format suitable for LLM APIs.
        
        For messages with tool_calls (assistant messages), formats tool calls according to
        the Ollama/OpenAI standard with a 'function' wrapper.
        """
        result = {
            "role": self.role,
            "content": self.content,
        }
        
        # Only include thinking if it's not empty
        if self.thinking:
            result["thinking"] = self.thinking
            
        # Format tool_calls with the 'function' wrapper for API compatibility
        if self.tool_calls:
            result["tool_calls"] = []
            for tc in self.tool_calls:
                tool_call_dict = {
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments
                    }
                }
                if tc.id is not None:
                    tool_call_dict["id"] = tc.id
                result["tool_calls"].append(tool_call_dict)
        
        return result

    def __str__(self):
        return f"Message(role={self.role}, content={self.content}, thinking={self.thinking}, tool_calls={self.tool_calls})"