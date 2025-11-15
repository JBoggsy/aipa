from agents.agent import Agent
from email_handling.email_objects import EmailThread, EmailMessage
from models.model import Model


class EmailAgent(Agent):
    def __init__(self, model: Model, agent_context=None):
        super().__init__(model, prompt_dir="agents/prompts/email_agent", agent_context=agent_context)
        self.email_sort_prompt = self.prompt_set["email_sort_prompt"]

    def process_email(self, email: EmailMessage) -> str:
        """
        Process a single email.
        
        Args:
            email: An EmailMessage object to process
            
        Returns:
            A string response from the LLM.
        """
        # Format the email processing prompt with the email details
        user_prompt = self.prompt_set["email_process_prompt"](email=email.as_formatted_string())
        
        # Generate messages and get LLM response
        prompt_messages = self.make_initial_prompt(user_prompt)
        response_messages = self.generate(prompt_messages, reasoning=False)

        return response_messages[0].content.strip()

    def summarize_email(self, email: EmailMessage) -> str:
        """
        Summarizes a single email using the LLM.
        
        Args:
            email: An EmailMessage object to summarize
            
        Returns:
            A string summary of the email.
        """
        # Format the email summary prompt with the email details
        user_prompt = self.prompt_set["email_summary_prompt"](email=email.as_formatted_string())
        
        # Generate messages and get LLM response
        prompt_messages = self.make_initial_prompt(user_prompt)
        response_messages = self.generate(prompt_messages, reasoning=False)

        return response_messages[0].content.strip()

    def sort_threads(self, threads: list[EmailThread]) -> list[str]:
        """
        Sorts email threads into categories based on the first email in each thread.
        
        Args:
            threads: List of EmailThread objects to categorize
            
        Returns:
            List of category labels, one per thread. Returns None for empty threads.
        """
        categories = []
        
        for thread in threads:
            if not thread.messages:
                categories.append(None)
                continue
            
            # Get the first email in the thread
            first_email = thread.messages[0]
            
            # Format the email sort prompt with the email details
            user_prompt = self.email_sort_prompt(email=first_email.as_formatted_string())
            
            # Generate messages and get LLM response
            messages = self.make_initial_prompt(user_prompt)
            message = self.model.generate(messages, reasoning=False)
            
            # Parse and validate the category
            category = self._parse_category(message.content)
            categories.append(category)
        
        return categories

    def _parse_category(self, response: str) -> list[str]:
        """
        Parse and validate the category from the LLM response.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            A list of validated category labels.
        """
        # Clean up the response
        cleaned_response = response.strip().upper()

        # Separate comma-separated categories and validate each
        category_list = [cat.strip() for cat in cleaned_response.split(",")]

        return category_list