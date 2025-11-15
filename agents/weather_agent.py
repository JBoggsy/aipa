from datetime import datetime
import json

from agents.agent import Agent
from agents.agent_context import AgentContext
from models.model import Model
from utils import get_geolocation, get_weather_data


class WeatherAgent(Agent):
    def __init__(self, model: Model, prompt_dir="agents/prompts/weather_agent", agent_context: AgentContext | None = None):
        super().__init__(model, prompt_dir, agent_context)

    def agent_as_tool(self) -> callable:
        """
        Return this agent as a tool callable.
        
        Returns:
            callable: A function that can be added as a tool to another agent.
        """
        def gen_morning_report() -> str:
            """
            Generate a morning weather report based on current and daily weather data.

            Returns:
                str: The generated morning weather report.
            """
            return self._gen_morning_report()
        return gen_morning_report

    def _gen_morning_report(self: 'WeatherAgent') -> str:
        """
        Generate a morning weather report based on current and daily weather data.

        Returns:
            str: The generated morning weather report.
        """
        geolocation = get_geolocation()
        lat = geolocation["lat"]
        long = geolocation["lng"]

        weather_data = get_weather_data((lat, long))
        current_weather = json.dumps(weather_data["current"], indent=2)
        daily_weather_data = json.dumps(weather_data["daily"][0], indent=2)

        now = datetime.now()
        current_time = now.strftime("%I:%M %p")

        user_prompt = self.prompt_set["morning_report_prompt"](
            current_time=current_time,
            current_weather=current_weather,
            daily_weather=daily_weather_data
        )

        messages = self.make_initial_prompt(user_prompt)
        message = self.model.generate(messages)
        return message.content