from ollama import chat

from models.model import Model
from messages import Message, ToolCall


class OllamaModel(Model):
    def __init__(self, model_name: str, **model_kwargs):
        super().__init__()
        self.model_name = model_name

    def generate(self, messages: list[Message],
                 max_length: int = 2048,
                 temperature: float = 0.8,
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
        # Convert Message objects to dictionaries for the Ollama API
        message_dicts = [msg.to_dict() for msg in messages]
        
        chat_kwargs = {
            "model": self.model_name,
            "messages": message_dicts,
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
        tool_calls = self.parse_tool_calls(message.tool_calls)
        
        return Message(
            role=message.role,
            content=message.content if message.content else "",
            thinking=message.thinking if hasattr(message, 'thinking') and message.thinking else "",
            tool_calls=tool_calls
        )

    def parse_tool_calls(self, raw_tool_calls) -> list[ToolCall] | None:
        """
        Parse tool calls from Ollama's raw response format into ToolCall instances.
        
        Args:
            raw_tool_calls: The raw tool call data from Ollama
            
        Returns:
            list[ToolCall] | None: A list of ToolCall instances, or None if no tool calls
        """
        if not raw_tool_calls:
            return None
            
        tool_calls = []
        for tc in raw_tool_calls:
            # Ollama tool calls have 'function' attribute with 'name' and 'arguments'
            if hasattr(tc, 'function'):
                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                    id=tc.id if hasattr(tc, 'id') else None
                ))
            # Handle dict format if Ollama returns it that way
            elif isinstance(tc, dict):
                if 'function' in tc:
                    tool_calls.append(ToolCall(
                        name=tc['function']['name'],
                        arguments=tc['function']['arguments'],
                        id=tc.get('id')
                    ))
                else:
                    # Fallback for simple dict format
                    tool_calls.append(ToolCall(
                        name=tc.get('name', ''),
                        arguments=tc.get('arguments', {}),
                        id=tc.get('id')
                    ))
        
        return tool_calls if tool_calls else None

    def add_tool(self, tool_schema: dict, tool_function: callable):
        tool_name = tool_schema["name"]
        self.tools[tool_name] = {
            "tool_dict": tool_schema,
            "function": tool_function
        }