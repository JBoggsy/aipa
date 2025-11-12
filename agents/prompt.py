from pathlib import Path
from jinja2 import Template


class Prompt:
    def __init__(self, prompt_str: str):
        self.prompt_str = prompt_str
        self.template = Template(prompt_str)

    def __call__(self,**kwargs) -> str:
        return self.template.render(**kwargs)
    
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
    def __init__(self, prompt_dirs: str | list[str]):
        """
        Initialize a PromptSet from one or more prompt directories.
        
        Args:
            prompt_dirs: A single directory path or a list of directory paths.
                        Prompts are loaded from all directories, with later
                        directories taking precedence over earlier ones if there
                        are duplicate prompt names.
        """
        if isinstance(prompt_dirs, str):
            prompt_dirs = [prompt_dirs]
        self.prompt_dirs = [Path(d) for d in prompt_dirs]
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict[str, str]:
        """Load prompts from all directories, with later dirs overriding earlier ones."""
        prompts = {}
        for prompt_dir in self.prompt_dirs:
            if not prompt_dir.exists():
                continue
            for prompt_file in prompt_dir.glob("*.txt"):
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