from agents.weather_agent import WeatherAgent
from agents.assistant_agent import AssistantAgent


if __name__ == "__main__": 
    assistant_agent = AssistantAgent("HuggingFaceTB/SmolLM3-3B", "agents/prompts/assistant_agent")
    assistant_agent.user_context.add_context("RECURRING INSTRUCTION: Wake me up at 7:00 AM every weekday.")
    assistant_agent.user_context.add_context("SCHEDULE: Video call at 10:00 AM about project X. Remidnder has been set.")
    assistant_agent.user_context.add_context("USER PREFERENCE: User prefers coffee in the morning.")
    assistant_agent.user_context.add_context("SCHEDULE: User will be going to the gym at 2:00 PM. Reminder has been set.")

    tasks_message = assistant_agent.gen_assistant_tasks()
    print("Assistant Tasks:")
    print(tasks_message)    