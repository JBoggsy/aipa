from datetime import datetime

from agents.agent import Agent
from agents.email_agent import EmailAgent
from agents.weather_agent import WeatherAgent
from email_handling.gmail_handler import GMAIL_HANDLER
from messages.message import Message
from models import Model, OllamaModel
from tasks import Task
from tools import get_current_time, get_user_location, get_weather_data
from utils import get_geolocation


class AssistantAgent(Agent):
    def __init__(self, model: Model, prompt_dir = "agents/prompts/assistant_agent"):
        super().__init__(model, prompt_dir)
        self.email_handler_agent = EmailAgent(OllamaModel("gpt-oss:20b"), agent_context=self.agent_context)
        self.weather_agent = WeatherAgent(OllamaModel("gpt-oss:20b"), agent_context=self.agent_context)

        self.add_tool(self.weather_agent.agent_as_tool())
        self.add_tool(get_current_time)
        self.add_tool(get_user_location)
        self.add_tool(get_weather_data)

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
        self.add_tool(mark_task_completed)

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