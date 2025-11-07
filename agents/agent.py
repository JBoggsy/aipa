import json
import re
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from agents.prompt_set import PromptSet, Prompt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Agent:
    LOADED_TOKENIZERS = {}
    LOADED_MODELS = {}

    def __init__(self, model_name: str, prompt_dir: str = "agents/prompts"):
        self.model_name = model_name
        self.prompt_set = PromptSet(prompt_dir)
        self.system_prompt = self.prompt_set["system_prompt"]()
        self.secrets = self.load_secrets()
        self.tools = {}

    @property
    def tokenizer(self):
        if self.model_name not in Agent.LOADED_TOKENIZERS:
            Agent.LOADED_TOKENIZERS[self.model_name] = AutoTokenizer.from_pretrained(self.model_name)
        return Agent.LOADED_TOKENIZERS[self.model_name]
    
    @property
    def model(self):
        if self.model_name not in Agent.LOADED_MODELS:
            Agent.LOADED_MODELS[self.model_name] = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=DEVICE
            )
        return Agent.LOADED_MODELS[self.model_name]

    @property
    def tool_dicts(self) -> list:
        return [tool["tool_dict"] for tool in self.tools.values()]
        
    def load_secrets(self) -> dict:
        with open("config/secrets.json", "r") as file:
            return json.load(file)

    def add_tool(self, name: str, description: str, parameters: dict, function: callable):
        self.tools[name] = {
            "tool_dict": {
                "name": name,
                "description": description,
                "parameters": parameters
            },
            "function": function
        }

    def make_simple_messages(self, user_prompt: str) -> list:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def parse_tool_calls(self, response: str) -> list:
        tool_call_pattern = re.compile(r"\<tool_call\>(.*?)\<\/tool_call\>", re.DOTALL)
        matches = tool_call_pattern.findall(response)
        tool_calls = []
        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue

        response = tool_call_pattern.sub("", response).strip()
        return tool_calls, response
    
    def execute_tool_call(self, tool_call: dict) -> str:
        tool_name = tool_call["name"]
        parameters = tool_call["arguments"]
        if tool_name in self.tools:
            tool_function = self.tools[tool_name]["function"]
            return tool_function(**parameters)
        else:
            raise ValueError(f"Tool '{tool_name}' not found.")

    def split_thinking(self, response: str) -> tuple[str, str]:
        thinking_pattern = re.compile(r"\<think\>(.*?)\<\/think\>", re.DOTALL)
        match = thinking_pattern.search(response)
        if match:
            thinking = match.group(1).strip()
            rest = thinking_pattern.sub("", response).strip()
            return thinking, rest
        return "", response

    def generate_response(self, 
                          messages: list, 
                          max_length: int = 2048, 
                          tool_use=True, 
                          think=True,
                          gen_kwargs={}) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            xml_tools=self.tool_dicts if tool_use else None,
            enable_thinking=think
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)
        output_tokens = self.model.generate(**model_inputs, 
                                            max_new_tokens=max_length,
                                            **gen_kwargs)[0]
        response_tokens = output_tokens[len(model_inputs.input_ids[0]):]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)