from datetime import datetime

from agents.agent import Agent
from agents.email_sorter_agent import EmailSorterAgent
from agents.user_descriptor_agent import UserDescriptorAgent
from agents.wakeup_agent import WakeupAgent
from agents.weather_agent import WeatherAgent

from email_handling.gmail_handler import GmailHandler
from models import Model, HFAutoModel, OllamaModel
from tasks import Task
from user import UserContext
from utils import get_geolocation


class AssistantAgent(Agent):
    def __init__(self, model: Model, prompt_dir = "agents/prompts/assistant_agent"):
        super().__init__(model, prompt_dir)
        self.user_context = UserContext()
        self.gmail_handler = GmailHandler()

        self.email_sorter_agent = EmailSorterAgent(OllamaModel("gpt-oss:20b"))
        self.wakeup_agent = WakeupAgent(OllamaModel("gpt-oss:20b"))
        self.weather_agent = WeatherAgent(OllamaModel("gpt-oss:20b"))

        user_descriptor_model = OllamaModel("gpt-oss:20b", format='json')
        self.user_descriptor_agent = UserDescriptorAgent(user_descriptor_model, self.user_context)

        self.add_tool(*self.weather_agent.agent_as_tool())
        self.add_tool(*self.wakeup_agent.agent_as_tool())
        self.add_tool({
            "name": "gen_daily_summary",
            "description": self.gen_daily_summary.__doc__,
            "parameters": {}
        }, self.gen_daily_summary)
        self.add_tool(Task.create_task_tool_schema(), Task.create_task)

    def gen_assistant_tasks(self) -> str:
        """
        Generates a list of tasks for the assistant based on the user's context.

        Returns:
            str: A list of tasks for the assistant.
        """
        now = datetime.now()
        timestamp = now.strftime("%I:%M %p on %A, %B %d, %Y")
        timestamp = "07:00 AM on Monday, November 5, 2025"  # For consistent testing

        geolocation = get_geolocation()
        location = f"{geolocation['city']}, {geolocation['state']}, {geolocation['country']}"

        user_prompt = self.prompt_set["agent_task_gen_prompt"](
            timestamp=timestamp,
            location=location,
            user_context=self.user_context.get_context()
        )

        messages = self.make_simple_messages(user_prompt)
        message = self.model.generate(messages, 
                                      max_length=4096,
                                      reasoning=True)

        if message.tool_calls is not None and len(message.tool_calls) > 0:
            tool_results = self.execute_tool_call(message.tool_calls)
            print("Tool Results:\n", tool_results)
        return message.content

    def gen_daily_summary(self: 'AssistantAgent') -> str:
        """
        Generates a summary of the user's day including events, tasks, and reminders.

        Returns:
            str: A summary of the user's day.
        """
        return "Placeholder for daily summary generation."