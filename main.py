from models import HFAutoModel, OllamaModel

from agents.agent import Agent
from agents.weather_agent import WeatherAgent
from agents.assistant_agent import AssistantAgent


if __name__ == "__main__": 
    assistant_model = OllamaModel("gpt-oss:20b")
    assistant_agent = AssistantAgent(assistant_model, "agents/prompts/assistant_agent")

    assistant_agent.user_context.add_context("RECURRING INSTRUCTION: Wake me up at 7:00 AM every weekday.")
    assistant_agent.user_context.add_context("SCHEDULE: Video call at 10:00 AM about project X. Reminder has been set.")
    assistant_agent.user_context.add_context("USER PREFERENCE: User prefers coffee in the morning.")
    assistant_agent.user_context.add_context("SCHEDULE: User will be going to the gym at 2:00 PM. Reminder has been set.")

    assistant_agent.cycle_step()
    assistant_agent.cycle_step()

    # assistant_agent.user_context.add_descriptive_statement(
    #     "User's name is James Boggs.", 1.0)
    # assistant_agent.user_context.add_descriptive_statement(
    #     "User's birthday is July 5, 1995.", 1.0)
    # assistant_agent.user_context.add_descriptive_statement(
    #     "User lives in Charlotte, NC", 1.0)
    # assistant_agent.user_context.add_descriptive_statement(
    #     "User is a male.", 1.0)
    # assistant_agent.user_context.add_descriptive_statement(
    #     "User works as a software developer.", 0.9)
    # response = assistant_agent.user_descriptor_agent.update_descriptive_statements([
    #     {
    #         "type": "email",
    #         "timestamp": "2024-06-01 09:15 AM",
    #         "content": "Message from Alice Jenkins in accounting: timesheets are due in a week."
    #     },
    #     {
    #         "type": "calendar_event",
    #         "timestamp": "2024-06-01 11:00 AM",
    #         "content": "Attended a meeting with the marketing team to discuss campaign strategies."
    #     },
    #     {
    #         "type": "email",
    #         "timestamp": "2024-06-01 03:30 PM",
    #         "content": "Sent a follow-up email to Bob regarding the budget proposal."
    #     },
    #     {
    #         "type": "email",
    #         "timestamp": "2024-06-02 08:45 AM",
    #         "content": "Emailed Jeff about playing new MMO game this weekend."
    #     },
    #     {
    #         "type": "calendar_event",
    #         "timestamp": "2024-06-02 01:00 PM",
    #         "content": "Lunch with Sarah at the new Italian restaurant downtown."
        
    #     }
    # ])
    # print("Updated Descriptive Statements:")
    # print(response)