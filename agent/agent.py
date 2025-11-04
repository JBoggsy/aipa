from datetime import datetime
import json
import requests
import re
import torch

import geocoder
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_geolocation


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Agent:
    def __init__(self, model_name: str, prompt_file: str = "agent/prompts.json"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        self.prompts = self.load_prompts(prompt_file)
        self.system_prompt = self.prompts["system_prompt"]
        self.secrets = self.load_secrets()

    def load_prompts(self, filepath: str) -> dict:
        with open(filepath, "r") as file:
            return json.load(file)
        
    def load_secrets(self) -> dict:
        with open("config/secrets.json", "r") as file:
            return json.load(file)

    def make_simple_messages(self, user_prompt: str) -> list:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def split_thinking(self, response: str) -> tuple[str, str]:
        thinking_pattern = re.compile(r"\<think\>(.*?)\<\/think\>", re.DOTALL)
        match = thinking_pattern.search(response)
        if match:
            thinking = match.group(1).strip()
            rest = thinking_pattern.sub("", response).strip()
            return thinking, rest
        return "", response

    def generate_response(self, messages: list, max_length: int = 2048) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)
        output_tokens = self.model.generate(**model_inputs, max_new_tokens=32768)[0]
        response_tokens = output_tokens[len(model_inputs.input_ids[0]):]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        

class WeatherAgent(Agent):
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
    
    def gen_morning_report(self) -> str:
        geolocation = get_geolocation()
        lat = geolocation["lat"]
        long = geolocation["lng"]

        weather_data = self.get_weather_data(lat, long)
        current_weather = json.dumps(weather_data["current"], indent=2)
        daily_weather_data = json.dumps(weather_data["daily"][0], indent=2)

        now = datetime.now()
        current_time = now.strftime("%I:%M %p")

        user_prompt = self.prompts["morning_report_prompt"].format(
            current_time=current_time,
            current_weather=current_weather,
            daily_weather_data=daily_weather_data
        )

        messages = self.make_simple_messages(user_prompt)
        response = self.generate_response(messages)
        thinking, response = self.split_thinking(response)
        return response
    

class AssistantAgent(Agent):
    def gen_morning_wakeup(self, morning_weather_report: str, daily_summary: str) -> str:
        now = datetime.now()
        current_time = now.strftime("%I:%M %p")
        current_date = now.strftime("%A, %B %d, %Y")

        geolocation = get_geolocation()
        lat = geolocation["lat"]
        long = geolocation["lng"]
        location = f"{geolocation['city']}, {geolocation['state']}, {geolocation['country']}"

        user_prompt = self.prompts["initial_wakeup_prompt"].format(
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
    

if __name__ == "__main__":
    weather_agent = WeatherAgent("HuggingFaceTB/SmolLM3-3B", "agent/weather_agent_prompts.json")
    morning_report = weather_agent.gen_morning_report()

    assistant_agent = AssistantAgent("HuggingFaceTB/SmolLM3-3B", "agent/assistant_prompts.json")
    wakeup_message = assistant_agent.gen_morning_wakeup(morning_report, "Placeholder for daily summary")
    print("Wakeup Message:")
    print(wakeup_message)