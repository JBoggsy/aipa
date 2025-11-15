from datetime import datetime

from agents.agent import Agent
from agents.agent_context import AgentContext
from models.model import Model
from utils import get_geolocation


class WakeupAgent(Agent):
    def __init__(self, model: Model, prompt_dir="agents/prompts/wakeup_agent", agent_context: AgentContext | None = None):
        super().__init__(model, prompt_dir, agent_context)

    def agent_as_tool(self) -> dict:
        def morning_wakeup(morning_weather_report: str, daily_summary: str) -> str:
            """
            Activates alarm and reads a morning wakeup for the user.

            This function activates a wakeup alarm on the user's device and then
            generates and reads aloud a morning wakeup message. The wakeup message
            prepares the user for the day ahead.

            Args:
                morning_weather_report (str): A brief report of the current and forecasted weather.
                daily_summary (str): A summary of the day's events and tasks.

            Returns:
                str: The generated morning wakeup message.
            """
            return self._morning_wakeup(morning_weather_report, daily_summary)
        
        schema = {
            "name": "morning_wakeup",
            "description": morning_wakeup.__doc__,    
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
        tool_func = morning_wakeup
        return schema, tool_func

    def _morning_wakeup(self: 'WakeupAgent', morning_weather_report: str, daily_summary: str) -> str:
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

        messages = self.make_initial_prompt(user_prompt)
        response_messages = self.generate(messages, max_length=2048, reasoning=True)

        print("Alarm activated for wakeup.")
        self.agent_context.add_context("ACTION TAKEN: Wakeup alarm activated.")

        print(f"Speaking message aloud: {response_messages[-1].content.strip()}")
        self.agent_context.add_context("ACTION TAKEN: Wakeup message spoken aloud.")

        return response_messages[-1].content.strip()