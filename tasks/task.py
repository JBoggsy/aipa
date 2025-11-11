class Task:
    """
    Represents a particular task an agent is undertaking.

    A `Task` is initially defined by a particular (small, achievable) goal that an agent wants to
    achieve. The agent then fleshes out the `Task` by generating a plan to accomplish the task and
    tests for whether each step is completed. Finally, the agent actually performs the task by
    executing the steps in the plan. The `Task` class stores all of the data and knowledge related
    to a particular task, including the goal, the plan, and a log of messages generated in carrying
    out the task.
    """
    
    def __init__(self, goal: str):
        self.goal = goal
        self.plan: str = ""
        self.message_log = []

    def add_plan(self, plan: str):
        self.plan = plan

    def log_message(self, role: str, content: str):
        self.message_log.append({"role": role, "content": content})

    @staticmethod
    def create_task(goal: str) -> "Task":
        """
        Create a new task with the specified goal.

        Args:
            goal (str): The goal of the task.

        Returns:
            Task: A new Task instance with the specified goal.
        """
        print(f"Creating new task with goal: {goal}")
        return Task(goal)
    
    @staticmethod
    def create_task_tool_schema() -> dict:
        """
        Returns the tool schema for creating a new task.

        Returns:
            dict: The tool schema for the create_task tool.
        """
        return {
            "name": "create_task",
            "description": "Creates a new task with a specified goal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "The goal of the new task."
                    }
                },
                "required": ["goal"]
            }
        }