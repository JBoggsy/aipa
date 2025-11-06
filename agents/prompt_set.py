from pathlib import Path


class Prompt:
    def __init__(self, prompt_str: str):
        self.prompt_str = prompt_str

    def __call__(self,**kwargs) -> str:
        return self.prompt_str.format(**kwargs)
    
    def __str__(self) -> str:
        return self.prompt_str
    
    def __len__(self) -> int:
        return len(self.prompt_str)
    
    def __repr__(self) -> str:
        return f"Prompt({self.prompt_str!r})"
    
    def __eq__(self, value):
        if isinstance(value, Prompt):
            return self.prompt_str == value.prompt_str
        return False

class PromptSet:
    def __init__(self, prompt_dir: str):
        self.prompt_dir = Path(prompt_dir)
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict[str, str]:
        prompts = {}
        for prompt_file in self.prompt_dir.glob("*.txt"):
            with open(prompt_file, "r") as f:
                prompt_name = prompt_file.stem
                prompts[prompt_name] = Prompt(f.read())
        return prompts
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, key: str) -> Prompt:
        return self.prompts[key]
    
    def __iter__(self):
        return iter(self.prompts.items())