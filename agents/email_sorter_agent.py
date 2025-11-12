from agents.agent import Agent
from email_handling.email_objects import EmailThread
from models.model import Model


VALID_CATEGORIES = ["ADVERTISEMENT", "EMAIL_BLAST", "BUSINESS", "PERSONAL"]


class EmailSorterAgent(Agent):
    def __init__(self, model: Model):
        super().__init__(model, prompt_dir="agents/prompts/email_sorter_agent")
        self.email_sort_prompt = self.prompt_set["email_sort_prompt"]

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
            user_prompt = self.email_sort_prompt(
                sender=first_email.sender,
                recipients=", ".join(first_email.recipients),
                subject=first_email.subject,
                body=first_email.body
            )
            
            # Generate messages and get LLM response
            messages = self.make_initial_prompt(user_prompt)
            message = self.model.generate(messages, reasoning=False)
            
            # Parse and validate the category
            category = self._parse_category(message.content)
            categories.append(category)
        
        return categories
    
    def _parse_category(self, response: str) -> str:
        """
        Parse and validate the category from the LLM response.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            A valid category string from VALID_CATEGORIES, or None if invalid
        """
        # Clean up the response
        cleaned_response = response.strip().upper()
        
        # Check if the response matches any valid category
        for category in VALID_CATEGORIES:
            if category in cleaned_response:
                return category
        
        # If no valid category found, return None
        return None
