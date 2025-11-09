

from models.model import Model


class OllamaModel(Model):
    def __init__(self, model_name: str, **model_kwargs):
        super().__init__()
        self.model = ChatOllama(
            model=model_name,
            validate_model_on_init=True,
            configurable_fields="any",
            **model_kwargs
        )

    def generate(self, messages: list,
                 max_length: int = 2048,
                 temperature: float = 0.8,
                 reasoning: bool = False) -> tuple[str, str, dict]:
        """
        Generates a response from the model based on the provided messages.

        Args:
            messages (list): A list of message dictionaries containing 'role' and 'content'.
            max_length (int, optional): The maximum length of the generated response. Defaults to 2048.
            temperature (float, optional): The sampling temperature for generation. Defaults to 0.8.
            reasoning (bool, optional): Whether to enable reasoning capabilities. Defaults to
            False

        Returns:
            tuple[str, str, dict]: A tuple containing the thinking process, the final response, and tool calls.
        """
        response_data = self.model.invoke(
            messages,
            config={
                "configurable": {
                    "num_predict": max_length,
                    "temperature": temperature,
                    "reasoning": reasoning
                }
            }
        )
        
        response = response_data.content
        thinking = response_data.additional_kwargs.get("reasoning_content", "")
        
        tool_results = {}
        if hasattr(response_data, "tool_calls"):
            tool_calls = response_data.tool_calls
        else:
            tool_calls = []
        for tool_call in tool_calls:
            tool_result = self.execute_tool_call(tool_call)
            tool_results[tool_call["name"]] = tool_result
        return thinking, response.strip(), tool_results

    def execute_tool_call(self, tool_call: dict) -> str:
        tool_name = tool_call["name"]
        parameters = tool_call["args"]
        if tool_name in self.tools:
            tool_function = self.tools[tool_name]["function"]
            return tool_function(**parameters)
        else:
            raise ValueError(f"Tool '{tool_name}' not found.")
    
    def add_tool(self, tool_schema: dict, tool_function: callable):
        self.tools[tool_schema["name"]] = {
            "tool_dict": tool_schema,
            "function": tool_function
        }
        self.model = self.model.bind_tools([tool["function"] for tool in self.tools.values()])