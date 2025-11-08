import json
import re
import torch

from agents.prompt import PromptSet, Prompt
from models import HFAutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Agent:
    AGENT_HUB = {}

    def __init__(self, model_name: str, prompt_dir: str = "agents/prompts"):
        self.model_name = model_name
        self.model = HFAutoModel(model_name)
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
        Agent.AGENT_HUB[agent_name] = self

    def make_simple_messages(self, user_prompt: str) -> list:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def add_tool(self, tool_schema: dict, tool_func: callable):
        self.model.add_tool(tool_schema, tool_func)

    def generate_response(self, 
                          messages: list, 
                          max_length: int = 2048,
                          reasoning: bool = False) -> tuple[str, str]:
        """
        Generates a response from the model based on the provided messages.

        Args:
            messages (list): A list of message dictionaries containing 'role' and 'content'.
            max_length (int, optional): The maximum length of the generated response. Defaults to 2048.
            reasoning (bool, optional): Whether to enable reasoning capabilities. Defaults to False

        Returns:
            tuple[str, str]: A tuple containing the thinking process and the final response.
        """
        return self.model.generate(messages, max_length=max_length, reasoning=reasoning)
