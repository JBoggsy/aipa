import json
from agents.agent import Agent
from models.model import Model
from models.ollama_model import OllamaModel
from user.user_context import UserContext, DescriptiveStatement


class UserDescriptorAgent(Agent):
    def __init__(self, model: Model, user_context: UserContext):
        super().__init__(model, prompt_dir="agents/prompts/user_descriptor_agent")
        self.description_summary_prompt = self.prompt_set["description_summary_prompt"]
        self.user_context = user_context

    def update_descriptive_statements(self, information_sources: list[dict]):
        """
        Update the user's descriptive statements based on new information
        sources.

        Args:
            information_sources: A list of dictionaries containing information
            about the user. Each dictionary should have keys 'type',
            'timestamp', and 'content'.
        """
        information_sources_block = self._format_information_sources_block(information_sources)
        descriptive_statements_block = self._format_descriptive_statements_block()
        
        user_prompt = self.description_summary_prompt(
            information_sources=information_sources_block,
            descriptive_statements=descriptive_statements_block
        )

        messages = self.make_simple_messages(user_prompt)
        message = self.model.generate(messages,
                                      max_length=4096,
                                      reasoning=True,
                                      format="json")
        descriptive_statements = json.loads(message.content)
        return descriptive_statements
    
    def _format_information_sources_block(self, information_sources: list[dict]) -> str:
        sources_block = ""
        for i, source in enumerate(information_sources):
            if i > 0:
                sources_block += "\n\n---\n\n"
            sources_block += f"{source['type']}\n{source['timestamp']}\n{source['content']}"
        return sources_block.strip()
    
    def _format_descriptive_statements_block(self) -> str:
        statements_block = ""
        for i, descriptive_statement in enumerate(self.user_context.descriptive_statements):
            if i > 0:
                statements_block += "\n\n---\n\n"
            statements_block += f"{descriptive_statement.id}\n{descriptive_statement.content}\n{descriptive_statement.confidence}"
        return statements_block.strip()