from agents.weather_agent import WeatherAgent
from agents.assistant_agent import AssistantAgent


if __name__ == "__main__": 
    assistant_agent = AssistantAgent("HuggingFaceTB/SmolLM3-3B", "agents/prompts/assistant_agent")
    assistant_agent.user_context.add_context("RECURRING INSTRUCTION: Wake me up at 7:00 AM every weekday.")
    assistant_agent.user_context.add_context("SCHEDULE: Video call at 10:00 AM about project X. Remidnder has been set.")
    assistant_agent.user_context.add_context("USER PREFERENCE: User prefers coffee in the morning.")
    assistant_agent.user_context.add_context("SCHEDULE: User will be going to the gym at 2:00 PM. Reminder has been set.")

    # tasks_message = assistant_agent.gen_assistant_tasks()
    # print("Assistant Tasks:")
    # print(tasks_message)    

    response = assistant_agent.user_descriptor_agent.update_descriptive_statements([
        {
            "type": "email",
            "timestamp": "2024-06-01 09:15 AM",
            "content": "Received an email from Alice about the upcoming project deadline."
        },
        {
            "type": "calendar_event",
            "timestamp": "2024-06-01 11:00 AM",
            "content": "Attended a meeting with the marketing team to discuss campaign strategies."
        },
        {
            "type": "email",
            "timestamp": "2024-06-01 03:30 PM",
            "content": "Sent a follow-up email to Bob regarding the budget proposal."
        },
        {
            "type": "email",
            "timestamp": "2024-06-02 08:45 AM",
            "content": "Emailed Jeff about playing new MMO game this weekend."
        },
        {
            "type": "calendar_event",
            "timestamp": "2024-06-02 01:00 PM",
            "content": "Lunch with Sarah at the new Italian restaurant downtown."
        
        }
    ])
    print("Updated Descriptive Statements:")
    print(response)
    for statement in assistant_agent.user_context.descriptive_statements:
        print(f"ID: {statement.id}\nContent: {statement.content}\nConfidence: {statement.confidence}\n")