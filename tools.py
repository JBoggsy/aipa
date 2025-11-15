from datetime import datetime
import json

from utils import get_geolocation, get_weather_data
from email_handling import GMAIL_HANDLER


######################
# IRL Context Tools  #
######################

def get_user_location() -> str:
    """
    Get a string representation of the user's geolocation based on their IP address.

    Returns:
        str: A formatted string containing city, state, and country.
    """
    location = get_geolocation()
    return f"{location['city']}, {location['state']}, {location['country']}"


def get_current_time() -> str:
    """
    Get the current time formatted as a string.

    Returns:
        str: The current time in the format "HH:MM AM/PM on Day, Month Date, Year".
    """
    now = datetime.now()
    return now.strftime("%I:%M %p on %A, %B %d, %Y")


def get_weather_data() -> str:
    """
    Fetch weather data for the user's current geolocation.

    This tool determines the user's location based on their IP address and retrieves the
    corresponding weather data as a JSON object. The returned JSON object includes the current
    weather conditions and forecasted weather information for the day.

    Returns:
        str: The weather data retrieved from the API as a JSON string.
    """
    geolocation = get_geolocation()
    lat = geolocation["lat"]
    long = geolocation["lng"]
    weather_data = get_weather_data((lat, long))
    return json.dumps(weather_data, indent=2)


######################
# User Context Tools #
######################
def get_unread_emails(count: int = 5) -> str:
    """
    Fetch the user's most recent unread emails.

    This tool retrieves up to the specified number of unread emails from the user's email and
    provides them as a JSON list of message IDs.

    Args:
        count (int, optional): The maximum number of unread email summaries to retrieve. Defaults to
        5.

    Returns:
        str: A JSON object containing the unread emails.
    """
    unread_emails = GMAIL_HANDLER.get_unread_emails(count)
    emails_json = [email.message_id for email in unread_emails]
    return json.dumps(emails_json, indent=2)


def get_email(message_id: str) -> str:
    """
    Fetch a specific email by its message ID.

    This tool retrieves the email corresponding to the provided message ID and returns its
    details as a formatted string.

    Args:
        message_id (str): The unique identifier of the email to retrieve.

    Returns:
        str: The email details as a formatted string.
    """
    email = GMAIL_HANDLER.get_message(message_id)
    if email:
        return str(email)
    else:
        return "Email not found."


##########################
# User Interaction Tools #
##########################

def say(text: str) -> str:
    """
    Use text-to-speech to say the given text.

    Args:
        text (str): The text to be spoken.

    Returns:
        str: Confirmation message.
    """
    # TODO: Actually speak sound out loud
    print(f"[TTS] {text}")
    return "Spoken successfully."

def activate_alarm() -> str:
    """
    Activates an alarm sound on the user's device.

    This tool plays an alarm sound for up to 60 seconds or until stopped by the user.

    Returns:
        str: Confirmation message.
    """
    # TODO: Implement actual alarm activation logic
    print("[Alarm] Activated.")
    return "Alarm activated."


def activate_lights() -> str:
    """
    Activates the smart lights in the user's environment.

    This tool sends a command to connected smart lights to turn them on.

    Returns:
        str: Confirmation message.
    """
    # TODO: Implement actual smart lights activation logic
    print("[Smart Lights] Activated.")
    return "Smart lights activated."


def activate_coffee_machine() -> str:
    """
    Activates an IoT coffee machine to brew a cup of coffee.

    This tool sends a command to the connected coffee machine to start brewing. The coffee will take
    approximately 5 minutes to be ready.

    Returns:
        str: Confirmation message.
    """
    # TODO: Implement actual coffee machine activation logic
    print("[Coffee Machine] Brewing started.")
    return "Coffee machine activated and brewing started."