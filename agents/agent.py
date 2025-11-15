import os
import torch
from dotenv import load_dotenv

from agents.prompt import PromptSet
from agents.agent_context import AgentContext
from models import Model
from messages import Message, ToolCall
from tasks import Task
from utils import generate_tool_schema

# Load environment variables from .env file
load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Agent:
    AGENT_HUB = {}

    def __init__(self, model: Model, prompt_dir: str | list[str] | None = None, agent_context: AgentContext | None = None):
        """
        Initialize an Agent.
        
        Args:
            model: The model to use for generating responses.
            prompt_dir: Optional directory or list of directories for agent-specific prompts.
                       All agents automatically load from 'agents/prompts/common' first,
                       then from any specified prompt_dir(s).
        """
        self.model = model
        
        # Build list of prompt directories, starting with common
        prompt_dirs = ["agents/prompts/common"]
        
        # Add agent-specific directories if provided
        if prompt_dir is not None:
            if isinstance(prompt_dir, str):
                prompt_dirs.append(prompt_dir)
            else:
                prompt_dirs.extend(prompt_dir)
        
        self.prompt_set = PromptSet(prompt_dirs)
        self.system_prompt = self.prompt_set["system_prompt"]()
        self.agent_context = agent_context if agent_context is not None else AgentContext()
        self.tools = {}
        self.tasks = []
        self.register_agent(self.__class__.__name__)

    @property
    def tool_dicts(self) -> list:
        return [tool["tool_dict"] for tool in self.tools.values()]
        
    def register_agent(self, agent_name: str):
        """
        Registers the agent in the global AGENT_HUB.

        Registering the agent allows for easy retrieval and management of different agents within
        the system.

        Args:
            agent_name (str): The name to register the agent under.
        """
        Agent.AGENT_HUB[agent_name] = self

    def make_system_message(self) -> Message:
        """
        Creates a system message using the agent's system prompt.
        """
        return Message(role="system", content=self.system_prompt)

    def make_initial_prompt(self, user_prompt: str) -> list[Message]:
        """
        Creates the initial prompt messages for the agent, including the system message and user
        prompt.

        Args:
            user_prompt (str): The user's prompt to include.

        Returns:
            list[Message]: A list of Message objects representing the initial prompt.
        """
        return [
            self.make_system_message(),
            Message(role="user", content=user_prompt)
        ]

    def add_tool(self, tool_func: callable):
        """
        Adds a tool to the agent's toolset.

        The tool schema is automatically generated from the function's name, docstring, and type
        hints. The function must have:
        - A descriptive docstring (first paragraph becomes the tool description)
        - Type hints for all parameters (except 'self')
        - Optional parameter descriptions in the docstring Args section

        The generated schema follows the format:
        {
            "name": "function_name",
            "description": "Description from docstring",
            "parameters": {
                "param_name": {
                    "type": "json_type",
                    "description": "Parameter description"
                }
            }
        }

        Args:
            tool_func (callable): The function to add as a tool. Must have docstring and type hints.
        
        Raises:
            ValueError: If the function lacks a docstring or has missing type hints.
        """
        tool_schema = generate_tool_schema(tool_func)
        self.tools[tool_schema["name"]] = {
            "tool_dict": tool_schema,
            "function": tool_func
        }
        self.model.add_tool(tool_schema, tool_func)

    def remove_tool(self, tool_name: str):
        """
        Removes a tool from the agent's toolset.

        The tool is removed from both the agent and the underlying model.

        Args:
            tool_name (str): The name of the tool to remove.
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
        self.model.remove_tool(tool_name)

    def generate(self, 
                         messages: list[Message],
                         max_length: int = 2048,
                         temperature: float = 0.1,
                         reasoning: bool = False,
                         format: str | None = None) -> list[Message]:
        """
        Generates a response from the model and executes any tool calls in the response.

        This method wraps Model.generate and automatically executes any tool calls returned by the
        model. It returns all resulting messages including the model's response and any tool call
        result messages.

        Args:
            messages (list[Message]): A list of Message objects to pass to the model.
            max_length (int, optional): The maximum length of the generated response. Defaults to 2048.
            temperature (float, optional): The sampling temperature for generation. Defaults to 0.8.
            reasoning (bool, optional): Whether to enable reasoning capabilities. Defaults to False.
            format (str | None, optional): The output format for the response. Defaults to None.

        Returns:
            list[Message]: A list of Message objects including the model's response and any tool
            call result messages.
        """
        # Generate response from the model
        response_message = self.model.generate(
            messages=messages,
            max_length=max_length,
            temperature=temperature,
            reasoning=reasoning,
            format=format
        )
        
        # Start with the model's response
        result_messages = [response_message]
        
        # Execute tool calls if any exist
        if response_message.tool_calls is not None and len(response_message.tool_calls) > 0:
            self.execute_tool_call(response_message.tool_calls)
            
            # Create a message for each tool call result
            for call in response_message.tool_calls:
                result_messages.append(call.to_message())

        return result_messages

    def execute_tool_call(self, tool_call: ToolCall | list[ToolCall]) -> list:
        """
        Executes one or more tool calls based on the provided tool call(s).

        This method a) calls the tool functions and b) populates the `result` attribute of the
        `ToolCall` objects. Therefore this method has any side effects that the provided `ToolCall`
        functions may have.
        
        Args:
            tool_call (ToolCall | list[ToolCall]): A ToolCall instance or list of ToolCall instances
            containing the tool name and arguments. Should be provided by the model's response.

        Returns:
            list: A list of results from executing each tool function. Always returns a list, even
            if only one tool call was provided.
        """
        tool_calls = tool_call if isinstance(tool_call, list) else [tool_call]
        for call in tool_calls:
            tool_name = call.name
            parameters = call.arguments
            if tool_name in self.tools:
                tool_function = self.tools[tool_name]["function"]
                call.result = tool_function(**parameters)
            else:
                raise ValueError(f"Tool '{tool_name}' not found.")