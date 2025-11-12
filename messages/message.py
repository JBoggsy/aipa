from dataclasses import dataclass
from typing import Optional, Any
import logging
from pathlib import Path
from datetime import datetime


@dataclass
class ToolCall:
    """
    Represents a tool call from an LLM.
    
    Attributes:
        name (str): The name of the tool to call
        arguments (dict): The arguments to pass to the tool function
        id (str | None): Optional identifier for the tool call (used by some LLM providers)
    """
    name: str
    arguments: dict
    result: Any = None
    id: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "arguments": self.arguments
        }
        if self.id is not None:
            result["id"] = self.id
        return result
    
    def to_message(self) -> 'Message':
        return Message(
            role="tool",
            content=f"{self.name}({self.arguments}):\n{self.result}"
        )

    def __str__(self):
        return f"ToolCall(name={self.name}, arguments={self.arguments}, result={self.result}, id={self.id})"

@dataclass
class Message:
    role: str
    content: str
    thinking: str = ""
    tool_calls: Optional[list['ToolCall']] = None
    
    # Class-level logger shared by all Message instances
    from typing import ClassVar
    _logger: ClassVar = logging.getLogger(__name__)
    _log_configured: ClassVar = False
    _log_lock: ClassVar = logging.Lock() if hasattr(logging, "Lock") else __import__("threading").Lock()
    _log_message_creation = True  # Set to False to disable logging of message creation

    @classmethod
    def set_message_creation_logging(cls, enabled: bool):
        """Enable or disable logging for Message creation."""
        cls._log_message_creation = enabled

    @classmethod
    def _configure_logging(cls):
        """Configure logging to write to a timestamped file in logs/ directory."""
        with cls._log_lock:
            if cls._log_configured:
                return

            # Create logs directory if it doesn't exist
            logs_dir = Path(__file__).parent.parent / "logs"
            logs_dir.mkdir(exist_ok=True)

            # Create timestamped log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = logs_dir / f"run_{timestamp}.log"

            # Configure file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            cls._logger.addHandler(file_handler)
            cls._logger.setLevel(logging.INFO)
            cls._log_configured = True

    def __post_init__(self):
        """Log message creation after dataclass initialization."""
        # Ensure logging is configured on first message creation
        self._configure_logging()

        if self._log_message_creation:
            tool_info = ""
            if self.tool_calls:
                tool_names = [tc.name for tc in self.tool_calls]
                tool_info = f" [tools: {', '.join(tool_names)}]"

            thinking_info = ""
            if self.thinking:
                thinking_info = f" [thinking: {self.thinking}]"

            self._logger.info(
                f"Message created - role: {self.role}, content:\n{self.content}\n{thinking_info}\n{tool_info}"
            )
        self._logger.info(
            f"Message created - role: {self.role}, content:\n{self.content}\n{thinking_info}\n{tool_info}"
        )

    def to_dict(self) -> dict:
        """
        Convert Message to dictionary format suitable for LLM APIs.
        
        For messages with tool_calls (assistant messages), formats tool calls according to
        the Ollama/OpenAI standard with a 'function' wrapper.
        """
        result = {
            "role": self.role,
            "content": self.content,
        }
        
        # Only include thinking if it's not empty
        if self.thinking:
            result["thinking"] = self.thinking
            
        # Format tool_calls with the 'function' wrapper for API compatibility
        if self.tool_calls:
            result["tool_calls"] = []
            for tc in self.tool_calls:
                tool_call_dict = {
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments
                    }
                }
                if tc.id is not None:
                    tool_call_dict["id"] = tc.id
                result["tool_calls"].append(tool_call_dict)
        
        return result

    def __str__(self):
        return f"Message(role={self.role}, content={self.content}, thinking={self.thinking}, tool_calls={self.tool_calls})"