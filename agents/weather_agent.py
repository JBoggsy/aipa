from datetime import datetime
import json
import requests

from agents.agent import Agent
from models.model import Model
from utils import get_geolocation


class WeatherAgent(Agent):
    def __init__(self, model: Model, prompt_dir="agents/prompts/weather_agent"):
        super().__init__(model, prompt_dir)

    def agent_as_tool(self) -> dict:
        schema = {
            "name": "gen_morning_report",
            "description": self.gen_morning_report.__doc__,    
            "parameters": {}
        }
        tool_func = self.gen_morning_report
        return schema, tool_func

    def post_process_weather_data(self, weather_data):
        """
        Convert UNIX timestamps in the weather data to human-readable datetime strings.
    
        Args:
            weather_data (dict or list): The weather data containing UNIX timestamps.

        Returns:
            dict or list: The weather data with converted datetime strings.
        """
        def _parse_timestamps(data):
            for k, v in data.items():
                if k in ["dt", "sunrise", "sunset", "moonrise", "moonset"]:
                    dt = datetime.fromtimestamp(v)
                    data[k] = str(dt)
        if isinstance(weather_data, list):
            for d in weather_data:
                _parse_timestamps(d)
        else:
            _parse_timestamps(weather_data)
        return weather_data

    def get_weather_data(self, lat: float, long: float) -> dict:
        """
        Fetch weather data from the OpenWeatherMap API.

        Args:
            lat (float): Latitude of the location.
            long (float): Longitude of the location.

        Returns:
            dict: The weather data retrieved from the API.
        """
        api_key = self.secrets["openweather_api_key"]
        url = f"https://api.openweathermap.org/data/3.0/onecall"
        params = {
            "lat": lat,
            "lon": long,
            "exclude": "minutely,hourly,alerts",
            "appid": api_key,
            "units": "imperial"
        }
        response = requests.get(url, params=params)

        if response.status_code != 200:
            # TODO: Handle error appropriately
            raise Exception(f"Error fetching weather data: {response.status_code}")
        weather_data = response.json()
        weather_data = self.post_process_weather_data(weather_data)
        return weather_data

    def gen_morning_report(self: 'WeatherAgent') -> str:
        """
        Generate a morning weather report based on current and daily weather data.

        Returns:
            str: The generated morning weather report.
        """
        geolocation = get_geolocation()
        lat = geolocation["lat"]
        long = geolocation["lng"]

        weather_data = self.get_weather_data(lat, long)
        current_weather = json.dumps(weather_data["current"], indent=2)
        daily_weather_data = json.dumps(weather_data["daily"][0], indent=2)

        now = datetime.now()
        current_time = now.strftime("%I:%M %p")

        user_prompt = self.prompt_set["morning_report_prompt"](
            current_time=current_time,
            current_weather=current_weather,
            daily_weather=daily_weather_data
        )

        messages = self.make_simple_messages(user_prompt)
        thinking, response, tool_results = self.generate_response(messages)
        return response