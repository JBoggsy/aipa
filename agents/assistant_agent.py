from datetime import datetime

from agents.agent import Agent
from agents.weather_agent import WeatherAgent
from user import UserContext
from utils import get_geolocation


class AssistantAgent(Agent):
    def __init__(self, model_name, prompt_dir = "agents/prompts/assistant_agent"):
        super().__init__(model_name, prompt_dir)
        self.user_context = UserContext()

        self.weather_agent = WeatherAgent(self.model_name, "agents/prompts/weather_agent")
        self.add_tool(
            name="gen_morning_wakeup",
            description="Activates a wakeup alarm on the user's device and then generates and reads aloud a morning wakeup message. The wakeup message prepares the user for the day ahead.",
            parameters={
                "daily_summary": {
                    "type": "string",
                    "description": "An executive summary of the day's events and tasks, focusing on scheduled events and the user's goals."
                }
            },
            function=self.gen_morning_wakeup
        )

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
        response = self.generate_response(messages, max_length=2048, tool_use=True, think=False)
        thinking, response = self.split_thinking(response)
        tool_calls, response = self.parse_tool_calls(response)

        if len(tool_calls) > 0:
            for tool_call in tool_calls:
                tool_response = self.execute_tool_call(tool_call)
                response += tool_response + "\n"
        return response.strip()

    def gen_morning_wakeup(self, daily_summary: str="Placeholder daily summary") -> str:
        """
        Generates a morning wakeup for the user.

        This function activates a wakeup alarm on the user's device and then
        generates and reads aloud a morning wakeup message. The wakeup message
        prepares the user for the day ahead.

        Args:
            daily_summary (str): A summary of the day's events and tasks.
        """
        now = datetime.now()
        current_time = now.strftime("%I:%M %p")
        current_date = now.strftime("%A, %B %d, %Y")
        current_time = "07:00 AM"  # For consistent testing
        current_date = "Monday, November 5, 2025"  # For consistent testing

        geolocation = get_geolocation()
        lat = geolocation["lat"]
        long = geolocation["lng"]
        location = f"{geolocation['city']}, {geolocation['state']}, {geolocation['country']}"

        morning_weather_report = self.weather_agent.gen_morning_report(lat, long)

        user_prompt = self.prompt_set["initial_wakeup_prompt"](
            current_time=current_time,
            current_date=current_date,
            location=location,
            weather_description=morning_weather_report,
            daily_summary=daily_summary
        )

        messages = self.make_simple_messages(user_prompt)
        response = self.generate_response(messages, max_length=2048, tool_use=False, think=False)
        thinking, response = self.split_thinking(response)
        return response
