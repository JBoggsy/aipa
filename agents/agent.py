import json
import torch

from agents.prompt import PromptSet
from models import Model
from messages import Message

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Agent:
    AGENT_HUB = {}

    def __init__(self, model: Model, prompt_dir: str = "agents/prompts"):
        self.model = model
        self.prompt_set = PromptSet(prompt_dir)
        self.system_prompt = self.prompt_set["system_prompt"]()
        self.secrets = self.load_secrets()
        self.tools = {}
        self.register_agent(self.__class__.__name__)

    @property
    def tool_dicts(self) -> list:
        return [tool["tool_dict"] for tool in self.tools.values()]
        
    def load_secrets(self) -> dict:
        with open("config/secrets.json", "r") as file:
            return json.load(file)
        
    def register_agent(self, agent_name: str):
        """
        Registers the agent in the global AGENT_HUB.

        Registering the agent allows for easy retrieval and management of different agents within
        the system.

        Args:
            agent_name (str): The name to register the agent under.
        """
        Agent.AGENT_HUB[agent_name] = self

    def make_simple_messages(self, user_prompt: str) -> list:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def add_tool(self, tool_schema: dict, tool_func: callable):
        """
        Adds a tool to the agent's toolset.

        The tool is registered both in the agent and in the underlying model. The `tool_schema`
        should follow the format:
        {
            "name": "tool_name",
            "description": "A description of what the tool does.",
            "parameters": {
                "parameter_name": {
                    "type": "parameter_type",
                    "description": "A description of the parameter."
                }
            }
        }

        Args:
            tool_schema (dict): A dictionary defining the tool's schema.
            tool_func (callable): The function that implements the tool's functionality.
        """
        self.tools[tool_schema["name"]] = {
            "tool_dict": tool_schema,
            "function": tool_func
        }
        self.model.add_tool(tool_schema, tool_func)

    def execute_tool_call(self, tool_call: dict) -> str:
        """
        Executes a tool call based on the provided tool call dictionary.
        
        Args:
            tool_call (dict): A dictionary containing the tool name and arguments. Should be
            provided by the model's response.

        Returns:
            str: The result of the tool function execution.
        """
        tool_name = tool_call["name"]
        parameters = tool_call["arguments"]
        if tool_name in self.tools:
            tool_function = self.tools[tool_name]["function"]
            return tool_function(**parameters)
        else:
            raise ValueError(f"Tool '{tool_name}' not found.")