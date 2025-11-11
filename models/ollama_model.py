from ollama import chat

from models.model import Model
from messages import Message


class OllamaModel(Model):
    def __init__(self, model_name: str, **model_kwargs):
        super().__init__()
        self.model_name = model_name

    def generate(self, messages: list,
                 max_length: int = 2048,
                 temperature: float = 0.8,
                 reasoning: bool = False,
                 format: str | None = None) -> Message:
        """
        Generates a response from the model based on the provided messages.

        Args:
            messages (list): A list of message dictionaries containing 'role' and 'content'.
            max_length (int, optional): The maximum length of the generated response. Defaults to 2048.
            temperature (float, optional): The sampling temperature for generation. Defaults to 0.8.
            reasoning (bool, optional): Whether to enable reasoning capabilities. Defaults to
            False
            format (str | None, optional): The output format for the response. Currently supports
            "json" for JSON-formatted output. Defaults to None.

        Returns:
            Message: A Message object containing the response, thinking process, and tool calls.
        """
        chat_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "options": {
                "num_predict": max_length,
                "temperature": temperature,
            },
            "think": reasoning,
            "tools": [tool["function"] for tool in self.tools.values()]
        }
        
        # Add format parameter if specified
        if format is not None:
            chat_kwargs["format"] = format
        
        response_data = chat(**chat_kwargs)

        message = response_data.message
        return Message(
            role=message.role,
            content=message.content if message.content else "",
            thinking=message.thinking if hasattr(message, 'thinking') and message.thinking else "",
            tool_calls=message.tool_calls if message.tool_calls else None
        )



    def add_tool(self, tool_schema: dict, tool_function: callable):
        tool_name = tool_schema["name"]
        self.tools[tool_name] = {
            "tool_dict": tool_schema,
            "function": tool_function
        }