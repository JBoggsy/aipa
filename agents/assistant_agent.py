from datetime import datetime

from agents.agent import Agent
from agents.email_agent import EmailAgent
from agents.wakeup_agent import WakeupAgent
from agents.weather_agent import WeatherAgent
from email_handling.gmail_handler import GmailHandler
from messages.message import Message
from models import Model, OllamaModel
from tasks import Task
from utils import get_geolocation


class AssistantAgent(Agent):
    def __init__(self, model: Model, prompt_dir = "agents/prompts/assistant_agent"):
        super().__init__(model, prompt_dir)

        self.gmail_handler = GmailHandler()
        self.email_handler_agent = EmailAgent(OllamaModel("gpt-oss:20b"), agent_context=self.agent_context)

        self.wakeup_agent = WakeupAgent(OllamaModel("gpt-oss:20b"), agent_context=self.agent_context)
        self.weather_agent = WeatherAgent(OllamaModel("gpt-oss:20b"), agent_context=self.agent_context)

        self.add_tool(*self.weather_agent.agent_as_tool())
        self.add_tool(*self.wakeup_agent.agent_as_tool())
        self.add_tool({
            "name": "gen_daily_summary",
            "description": self.get_gen_daily_summary_tool_func().__doc__,
            "parameters": {}
        }, self.get_gen_daily_summary_tool_func())
        self.add_tool({
            "name": "get_coffee",
            "description": self.get_coffee_tool_func().__doc__,
            "parameters": {}
        }, self.get_coffee_tool_func())
        self.add_tool({
            "name": "remove_notification",
            "description": self.get_remove_notification_tool_func().__doc__,
            "parameters": {
                "notification_id": {
                    "type": "integer",
                    "description": "The ID of the notification to remove."
                }
            }
        }, self.get_remove_notification_tool_func())
        self.add_tool({
            "name": "get_latest_email_summary",
            "description": self.get_latest_email_summary_tool_func().__doc__,
            "parameters": {}
        }, self.get_latest_email_summary_tool_func())

        self.debug_time = datetime(2025, 11, 5, 7, 0)  # Default debug time: Nov 5, 2025, 7:00 AM

    def explain_tools(self) -> str:
        """
        Generates an explanation of the assistant's available tools.

        Returns:
            str: A description of the assistant's tools.
        """
        user_prompt = "Write a short blurb explaining each of the tools you have access to."
        prompt_messages = self.make_initial_prompt(user_prompt)
        response_messages = self.generate(prompt_messages, 
                                      max_length=2048,
                                      temperature=0.5,
                                      reasoning=True)
        return response_messages[-1].content.strip()

    def cycle_step(self):
        """
        Executes a single cycle step for the assistant agent.

        If the agent has no tasks, it generates new tasks based on the user's context. Otherwise, it
        evaluates its current tasks and selects the one with the highest priority to execute next.
        """
        if len(self.tasks) == 0:
            self.gen_assistant_task()
        else:
            selected_task = self.select_next_task()
            self.execute_task(selected_task)

    def gen_assistant_task(self) -> None:
        """
        Generates a task for the assistant based on the user's context.
        """
        # now = datetime.now()
        # timestamp = now.strftime("%I:%M %p on %A, %B %d, %Y")
        timestamp = self.debug_time.strftime("%I:%M %p on %A, %B %d, %Y")

        geolocation = get_geolocation()
        location = f"{geolocation['city']}, {geolocation['state']}, {geolocation['country']}"

        user_prompt = self.prompt_set["agent_task_gen_prompt"](
            timestamp=timestamp,
            location=location,
            agent_context=self.agent_context.get_context(),
            agent_notifications=self.agent_context.get_notifications()
        )

        prompt_messages = self.make_initial_prompt(user_prompt)
        response_messages = self.generate(prompt_messages, 
                                          max_length=4096,
                                          reasoning=True)
        task_goal = response_messages[-1].content.strip()
        new_task = Task(goal=task_goal)
        self.tasks.append(new_task)

    def select_next_task(self) -> Task:
        """
        Selects the next task for the assistant to execute.

        Tasks do not have inherent priorities, since the importance of a task can vary based on
        context. Instead, the agent is given an enumerated list of its current tasks and selects one
        to execute next.

        Returns:
            Task: The selected task to execute next.
        """
        now = datetime.now()
        # timestamp = now.strftime("%I:%M %p on %A, %B %d, %Y")
        timestamp = self.debug_time.strftime("%I:%M %p on %A, %B %d, %Y")

        geolocation = get_geolocation()
        location = f"{geolocation['city']}, {geolocation['state']}, {geolocation['country']}"

        tasks_list = [{"goal": task.goal} for task in self.tasks]

        user_prompt = self.prompt_set["agent_task_selection_prompt"](
            timestamp=timestamp,
            location=location,
            agent_context=self.agent_context.get_context(),
            agent_tasks=tasks_list
        )

        prompt_messages = self.make_initial_prompt(user_prompt)
        response_messages = self.generate(prompt_messages, 
                                          max_length=4096,
                                          reasoning=True)

        # Parse the selected task index from the model's response
        try:
            selected_index = int(response_messages[-1].content.strip()) - 1
            if 0 <= selected_index < len(self.tasks):
                selected_task = self.tasks[selected_index]
                return selected_task
            else:
                raise ValueError("Selected index out of range.")
        except ValueError as e:
            raise Exception(f"Error parsing selected task index: {e}")
    
    def execute_task(self, task: Task) -> str:
        """
        Executes the given task.

        Args:
            task (Task): The task to execute.

        Returns:
            str: The result of the task execution.
        """
        if task.goal.lower().strip(".") == "standby":
            self.tasks.remove(task)
            return 
        
        mark_task_completed = self._mark_task_completed_func_factory(task)
        self.add_tool({
            "name": "mark_task_completed",
            "description": mark_task_completed.__doc__,
            "parameters": {}
        }, mark_task_completed)

        max_iterations = 20
        iterations = 0
        while not task.completed and iterations < max_iterations:
            if task.plan is None or len(task.plan) == 0:
                self.gen_task_plan(task)
            else:
                self.execute_task_step(task)
            iterations += 1
        self.remove_tool("mark_task_completed")
        if task.completed:
            self.agent_context.add_context(f"TASK COMPLETED: {task.goal}")
        self.tasks.remove(task)

    def _mark_task_completed_func_factory(self, task: Task) -> callable:
        """
        Creates the tool function for marking a task as completed.

        Args:
            task (Task): The task to mark as completed.
        
        Returns:
            callable: The mark_task_completed tool function.
        """
        def mark_task_completed() -> None:
            """Indicate that the task you are currently working on is completed."""
            task.completed = True
        return mark_task_completed

    def gen_task_plan(self, task: Task):
        """
        Generates a plan for the given task.

        Args:
            task (Task): The task to generate a plan for.

        Returns:
            str: A list of steps in the task plan.
        """
        # now = datetime.now()
        # timestamp = now.strftime("%I:%M %p on %A, %B %d, %Y")
        timestamp = self.debug_time.strftime("%I:%M %p on %A, %B %d, %Y")

        geolocation = get_geolocation()
        location = f"{geolocation['city']}, {geolocation['state']}, {geolocation['country']}"

        user_prompt = self.prompt_set["agent_task_planning_prompt"](
            timestamp=timestamp,
            location=location,
            agent_context=self.agent_context.get_context(),
            task_goal=task.goal
        )

        prompt_messages = self.make_initial_prompt(user_prompt)
        response_messages = self.generate(prompt_messages, 
                                        max_length=4096,
                                        reasoning=True)
        plan = response_messages[-1].content.strip()
        task.add_plan(plan)
        task.message_log.extend(prompt_messages)
        task.message_log.extend(response_messages)

    def execute_task_step(self, task: Task) -> str:
        """
        Executes a single step of the given task.

        Args:
            task (Task): The task to execute a step for.

        Returns:
            str: The result of the task step execution.
        """
        # now = datetime.now()
        # timestamp = now.strftime("%I:%M %p on %A, %B %d, %Y")
        timestamp = self.debug_time.strftime("%I:%M %p on %A, %B %d, %Y")

        geolocation = get_geolocation()
        location = f"{geolocation['city']}, {geolocation['state']}, {geolocation['country']}"

        user_prompt = self.prompt_set["agent_task_step_prompt"](
            timestamp=timestamp,
            location=location,
            agent_context=self.agent_context.get_context(),
            task=task
        )
        prompt_message = Message(role="user", content=user_prompt)

        messages = task.message_log.copy()
        messages.append(prompt_message)
        response_messages = self.generate(messages, 
                                      max_length=4096,
                                      reasoning=True)
        task.message_log.append(prompt_message)
        task.message_log.extend(response_messages)

    #########
    # TOOLS #
    #########

    def get_gen_daily_summary_tool_func(self) -> callable:
        """
        Creates the tool function for generating a daily summary.
        """
        def gen_daily_summary() -> str:
            """Generates a summary of the user's day including events, tasks, and reminders. 

            This tool does NOT interact with the user.

            Returns:
                str: A summary of the user's day.
            """
            return "Today, you have a video call about Project X at 10:00 AM and you're scheduled to go to lunch with Gabriel at 2:00pm. You also have a reminder to submit the quarterly report by the end of the day."
        return gen_daily_summary
    
    def get_coffee_tool_func(self) -> callable:
        """
        Creates the tool function for getting coffee.

        Returns:
            callable: The get_coffee tool function.
        """
        def get_coffee() -> int:
            """Activates the coffee maker to brew a fresh cup of coffee.

            This tool interacts with the user's coffee maker device.

            Returns:
                int: Return code indicating success (0) or failure (non-zero).
            """
            print("Brewing a fresh cup of coffee...")
            self.agent_context.add_context("ACTION TAKEN: Coffee brewed.")
            return 0
        return get_coffee
    
    def get_remove_notification_tool_func(self) -> callable:
        """
        Creates the tool function for removing a notification.

        Returns:
            callable: The remove_notification tool function.
        """
        def remove_notification(notification_id: int) -> str:
            """Removes a notification from your agentic context.

            This tool does NOT interact with the user. It removes a notification from your internal
            list of notifications based on the provided ID.

            Args:
                notification_id (int): The ID of the notification to remove.

            Returns:
                str: A message indicating the result of the removal.
            """
            if notification_id in self.agent_context.notifications:
                del self.agent_context.notifications[notification_id]
                return f"Removed notification {notification_id}"
            else:
                return f"Notification {notification_id} not found"
        return remove_notification
    
    def get_latest_email_summary_tool_func(self):
        """
        Creates the tool function for getting a summary of the latest email.

        Returns:
            callable: The get_latest_email_summary tool function.
        """
        def get_latest_email_summary() -> str:
            """Fetches and summarizes the latest email from the user's inbox.

            Returns:
                str: A summary of the latest email.
            """
            self.gmail_handler.update_emails()
            threads = list(self.gmail_handler.threads.values())
            if not threads:
                return "No email threads found."
            latest_thread = max(threads, key=lambda t: t.timestamp)
            if not latest_thread.messages:
                return "No messages in the latest thread."
            latest_email = latest_thread.messages[-1]
            summary = self.email_handler_agent.summarize_email(latest_email)
            return summary
        return get_latest_email_summary