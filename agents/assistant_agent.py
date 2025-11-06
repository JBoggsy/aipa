from datetime import datetime

from agents.agent import Agent
from user import UserContext
from utils import get_geolocation


class AssistantAgent(Agent):
    def __init__(self, model_name, prompt_dir = "agents/prompts/assistant_agent"):
        super().__init__(model_name, prompt_dir)
        self.user_context = UserContext()

    def gen_assistant_tasks(self) -> str:
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
        response = self.generate_response(messages)
        thinking, response = self.split_thinking(response)
        return response
    
    def gen_morning_wakeup(self, morning_weather_report: str, daily_summary: str) -> str:
        now = datetime.now()
        current_time = now.strftime("%I:%M %p")
        current_date = now.strftime("%A, %B %d, %Y")

        geolocation = get_geolocation()
        lat = geolocation["lat"]
        long = geolocation["lng"]
        location = f"{geolocation['city']}, {geolocation['state']}, {geolocation['country']}"

        user_prompt = self.prompt_set["initial_wakeup_prompt"](
            current_time=current_time,
            current_date=current_date,
            location=location,
            weather_description=morning_weather_report,
            daily_summary=daily_summary
        )

        messages = self.make_simple_messages(user_prompt)
        response = self.generate_response(messages)
        thinking, response = self.split_thinking(response)
        return response
