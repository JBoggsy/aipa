from datetime import datetime

from agents.agent import Agent
from agents.email_sorter_agent import EmailSorterAgent
from agents.user_descriptor_agent import UserDescriptorAgent
from agents.wakeup_agent import WakeupAgent
from agents.weather_agent import WeatherAgent
from messages.message import Message
from email_handling.gmail_handler import GmailHandler
from models import Model, HFAutoModel, OllamaModel
from tasks import Task
from user import UserContext
from utils import get_geolocation


class AssistantAgent(Agent):
    def __init__(self, model: Model, prompt_dir = "agents/prompts/assistant_agent"):
        super().__init__(model, prompt_dir)
        self.user_context = UserContext()
        self.gmail_handler = GmailHandler()

        self.email_sorter_agent = EmailSorterAgent(OllamaModel("gpt-oss:20b"))
        self.wakeup_agent = WakeupAgent(OllamaModel("gpt-oss:20b"))
        self.weather_agent = WeatherAgent(OllamaModel("gpt-oss:20b"))

        user_descriptor_model = OllamaModel("gpt-oss:20b", format='json')
        self.user_descriptor_agent = UserDescriptorAgent(user_descriptor_model, self.user_context)

        self.add_tool(*self.weather_agent.agent_as_tool())
        self.add_tool(*self.wakeup_agent.agent_as_tool())
        self.add_tool({
            "name": "gen_daily_summary",
            "description": self._gen_daily_summary.__doc__,
            "parameters": {}
        }, self.get_gen_daily_summary_tool_func())
        self.add_tool({
            "name": "get_coffee",
            "description": self._get_coffee.__doc__,
            "parameters": {}
        }, self.get_coffee_tool_func())

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
        now = datetime.now()
        timestamp = now.strftime("%I:%M %p on %A, %B %d, %Y")
        timestamp = "07:00 AM on Monday, November 5, 2025"  # For consistent testing

        geolocation = get_geolocation()
        location = f"{geolocation['city']}, {geolocation['state']}, {geolocation['country']}"

        user_prompt = self.prompt_set["agent_task_gen_prompt"](
            timestamp=timestamp,
            location=location,
            user_context=self.user_context.get_context()
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
        timestamp = now.strftime("%I:%M %p on %A, %B %d, %Y")
        timestamp = "07:00 AM on Monday, November 5, 2025"  # For consistent testing

        geolocation = get_geolocation()
        location = f"{geolocation['city']}, {geolocation['state']}, {geolocation['country']}"

        tasks_list = [{"goal": task.goal} for task in self.tasks]

        user_prompt = self.prompt_set["agent_task_selection_prompt"](
            timestamp=timestamp,
            location=location,
            user_context=self.user_context.get_context(),
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
        now = datetime.now()
        timestamp = now.strftime("%I:%M %p on %A, %B %d, %Y")
        timestamp = "07:00 AM on Monday, November 5, 2025"  # For consistent testing

        geolocation = get_geolocation()
        location = f"{geolocation['city']}, {geolocation['state']}, {geolocation['country']}"

        user_prompt = self.prompt_set["agent_task_planning_prompt"](
            timestamp=timestamp,
            location=location,
            user_context=self.user_context.get_context(),
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
        now = datetime.now()
        timestamp = now.strftime("%I:%M %p on %A, %B %d, %Y")
        timestamp = "07:00 AM on Monday, November 5, 2025"  # For consistent testing

        geolocation = get_geolocation()
        location = f"{geolocation['city']}, {geolocation['state']}, {geolocation['country']}"

        user_prompt = self.prompt_set["agent_task_step_prompt"](
            timestamp=timestamp,
            location=location,
            user_context=self.user_context.get_context(),
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

    def get_gen_daily_summary_tool_func(self) -> callable:
        """
        Creates the tool function for generating a daily summary.
        """
        def gen_daily_summary() -> str:
            """Generates a summary of the user's day including events, tasks, and reminders.

            Returns:
                str: A summary of the user's day.
            """
            return self._gen_daily_summary()
        return gen_daily_summary

    def _gen_daily_summary(self: 'AssistantAgent') -> str:
        """
        Generates a summary of the user's day including events, tasks, and reminders.

        Returns:
            str: A summary of the user's day.
        """
        return "Today, you have a video call about Project X at 10:00 AM and you're scheduled to go to lunch with Gabriel at 2:00pm. You also have a reminder to submit the quarterly report by the end of the day."
    
    def get_coffee_tool_func(self) -> callable:
        """
        Creates the tool function for getting coffee.

        Returns:
            callable: The get_coffee tool function.
        """
        def get_coffee() -> int:
            """Activates the coffee maker to brew a fresh cup of coffee.

            Returns:
                int: Return code indicating success (0) or failure (non-zero).
            """
            return self._get_coffee()
        return get_coffee

    def _get_coffee(self) -> int:
        """Activates the coffee maker to brew a fresh cup of coffee.

        Returns:
            int: Return code indicating success (0) or failure (non-zero).
        """
        return 0