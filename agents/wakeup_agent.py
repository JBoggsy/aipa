from datetime import datetime

from agents.agent import Agent
from utils import get_geolocation


class WakeupAgent(Agent):
    def __init__(self, model_name, prompt_dir="agents/prompts/wakeup_agent"):
        super().__init__(model_name, prompt_dir)

    def agent_as_tool(self) -> dict:
        schema = {
            "name": "get_morning_wakeup",
            "description": self.gen_morning_wakeup.__doc__,    
            "parameters": {
                "morning_weather_report": {
                    "type": "string",
                    "description": "A brief report of the current and forecasted weather."
                },
                "daily_summary": {
                    "type": "string",
                    "description": "A summary of the day's events and tasks."
                }
            }
        }
        tool_func = self.gen_morning_wakeup
        return schema, tool_func

    def gen_morning_wakeup(self: 'WakeupAgent', morning_weather_report: str, daily_summary: str) -> str:
        """
        Generates a morning wakeup for the user.

        This function activates a wakeup alarm on the user's device and then
        generates and reads aloud a morning wakeup message. The wakeup message
        prepares the user for the day ahead.

        Args:
            morning_weather_report (str): A brief report of the current and forecasted weather.
            daily_summary (str): A summary of the day's events and tasks.

        Returns:
            str: The generated morning wakeup message.
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

        user_prompt = self.prompt_set["initial_wakeup_prompt"](
            current_time=current_time,
            current_date=current_date,
            location=location,
            weather_description=morning_weather_report,
            daily_summary=daily_summary
        )

        messages = self.make_simple_messages(user_prompt)
        thinking, response = self.generate_response(messages, max_length=2048, reasoning=False)
        return response