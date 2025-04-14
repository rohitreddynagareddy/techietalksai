import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
import random

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (41 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}

def get_motivation(name: str) -> dict:
    """Generates a personalized motivational message.

    Args:
        name (str): The name of the person to include in the message.

    Returns:
        str: A randomly selected motivational message personalized with the given name.
    """

    messages = [
        "Live in the moment {}, you can do it!",
        "Stay strong {}, your best is yet to come!",
        "Keep pushing {}, success is within reach!",
        "Believe in yourself {}, greatness awaits!",
        "Never give up {}, every step counts!",
        "You're capable of amazing things, {}!",
        "One step at a time {}, keep going!",
        "You’ve got this {}, trust the process!",
        "Shine bright {}, the world needs your light!",
        "Keep moving forward {}, no matter what!"
    ]
    ans = random.choice(messages).format(name)
    report = (
        f'{ans}'
    )
    return {"status": "success", "report": report}

root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash-exp",
    description=(
        "Agent to answer questions about the time, motivation and weather in a city."
    ),
    instruction=(
        "I can answer your questions about the time, motivation and weather in a city."
    ),
    tools=[get_weather, get_current_time, get_motivation],
)
