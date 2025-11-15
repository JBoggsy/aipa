from requests_cache import datetime
from models import OllamaModel

from agents.assistant_agent import AssistantAgent

from email_handling import GmailHandler
from agents import EmailAgent


if __name__ == "__main__": 
    assistant_model = OllamaModel("gpt-oss:20b")
    assistant_agent = AssistantAgent(assistant_model, "agents/prompts/assistant_agent")

    assistant_agent.agent_context.add_context("RECURRING INSTRUCTION: Wake me up at 7:00 AM every weekday.")
    assistant_agent.agent_context.add_context("RECURRING INSTRUCTION: Have coffee ready by 7:15 AM every weekday.")

    assistant_agent.cycle_step()
    assistant_agent.cycle_step()

    assistant_agent.debug_time = datetime(2025, 11, 5, 7, 5)  # Nov 5, 2025, 7:05 AM
    
    assistant_agent.cycle_step()
    assistant_agent.cycle_step()

    assistant_agent.debug_time = datetime(2025, 11, 5, 7, 10)  # Nov 5, 2025, 7:10 AM
    assistant_agent.agent_context.add_notification("User received an email.")
    
    assistant_agent.cycle_step()
    assistant_agent.cycle_step()

    # gmail_handler = GmailHandler()
    # gmail_handler.update_emails()
    # email_agent = EmailAgent(OllamaModel("gpt-oss:20b"))
    # threads = list(gmail_handler.threads.values())[:5]  # Get first five threads
    # for thread in threads:
    #     summary = email_agent.summarize_email(thread.messages[0])
    #     print(f"Summary of first email in thread {thread.thread_id}:\n{summary}\n")