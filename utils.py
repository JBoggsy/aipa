from datetime import datetime
import inspect
import os
import requests
import geocoder
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def generate_tool_schema(func: callable) -> dict:
    """
    Generate a tool schema from a callable using its docstring and type hints.
    
    Args:
        func (callable): The function to generate a schema for.
        
    Returns:
        dict: A tool schema dictionary with 'name', 'description', and 'parameters'.
        
    Raises:
        ValueError: If the function lacks a docstring or has missing type hints for parameters.
    """
    # Get function name
    tool_name = func.__name__
    
    # Get description from docstring
    docstring = inspect.getdoc(func)
    if not docstring:
        raise ValueError(f"Function '{tool_name}' must have a docstring to generate tool schema")
    
    # Extract the first line or paragraph as description (before Args section)
    description_lines = []
    for line in docstring.split('\n'):
        line = line.strip()
        if line.startswith(('Args:', 'Returns:', 'Raises:', 'Example:')):
            break
        if line:
            description_lines.append(line)
    
    description = ' '.join(description_lines)
    
    # Get function signature
    sig = inspect.signature(func)
    parameters = {}
    
    for param_name, param in sig.parameters.items():
        # Skip self parameter
        if param_name == 'self':
            continue
            
        # Get type annotation
        if param.annotation == inspect.Parameter.empty:
            raise ValueError(f"Parameter '{param_name}' in function '{tool_name}' must have a type hint")
        
        # Convert Python type to JSON schema type
        param_type = param.annotation
        json_type = _python_type_to_json_type(param_type)
        
        # Try to extract parameter description from docstring
        param_description = _extract_param_description(docstring, param_name)
        
        parameters[param_name] = {
            "type": json_type,
            "description": param_description or f"The {param_name} parameter"
        }
    
    return {
        "name": tool_name,
        "description": description,
        "parameters": parameters
    }


def _python_type_to_json_type(python_type) -> str:
    """
    Convert Python type hints to JSON schema types.
    
    Args:
        python_type: The Python type annotation.
        
    Returns:
        str: The corresponding JSON schema type.
    """
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object"
    }
    
    # Handle basic types
    if python_type in type_mapping:
        return type_mapping[python_type]
    
    # Handle string representation (for type hints like 'str', 'int')
    type_str = str(python_type)
    if 'str' in type_str:
        return "string"
    elif 'int' in type_str:
        return "integer"
    elif 'float' in type_str:
        return "number"
    elif 'bool' in type_str:
        return "boolean"
    elif 'list' in type_str:
        return "array"
    elif 'dict' in type_str:
        return "object"
    
    # Default to string for unknown types
    return "string"


def _extract_param_description(docstring: str, param_name: str) -> str | None:
    """
    Extract parameter description from a docstring.
    
    Args:
        docstring (str): The function's docstring.
        param_name (str): The name of the parameter to extract description for.
        
    Returns:
        str | None: The parameter description if found, None otherwise.
    """
    if not docstring:
        return None
    
    lines = docstring.split('\n')
    in_args_section = False
    
    for i, line in enumerate(lines):
        if 'Args:' in line:
            in_args_section = True
            continue
        
        if in_args_section:
            # Check if we've left the Args section
            if line.strip() and not line.startswith(' ') and ':' in line:
                break
            
            # Check if this line contains the parameter
            if param_name in line and ':' in line:
                # Extract description after the colon
                parts = line.split(':', 1)
                if len(parts) == 2:
                    description = parts[1].strip()
                    # Handle multi-line descriptions
                    j = i + 1
                    while j < len(lines) and lines[j].startswith('        '):
                        description += ' ' + lines[j].strip()
                        j += 1
                    return description
    
    return None


def get_geolocation() -> dict:
    """
    Get the geolocation based on the user's IP address.

    Returns:
        dict: A dictionary containing geolocation information:
            - lat (float): Latitude
            - lng (float): Longitude
            - city (str): City name
            - state (str): State name
            - country (str): Country name
    """
    g = geocoder.ip('me')
    return {
        'lat': g.latlng[0],
        'lng': g.latlng[1],
        'city': g.city,
        'state': g.state,
        'country': g.country
    }


def post_process_weather_data(weather_data):
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
    

def get_weather_data(coords: tuple[float, float]|None = None) -> dict:
    """
    Fetch weather data from the OpenWeatherMap API.

    Args:
        coords (tuple[float, float]|None): A tuple containing latitude and longitude.
            If None, the geolocation will be determined based on the user's IP address.

    Returns:
        dict: The weather data retrieved from the API.
    """
    if coords is None:
        geolocation = get_geolocation()
        lat = geolocation["lat"]
        long = geolocation["lng"]
    else:
        lat, long = coords
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY not found in environment variables")
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
    weather_data = post_process_weather_data(weather_data)
    return weather_data