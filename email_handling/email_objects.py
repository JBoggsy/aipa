from base64 import urlsafe_b64decode
from dataclasses import dataclass, field


@dataclass
class EmailMessage:
    subject: str
    sender: str
    recipients: list[str]
    body: str
    labels: list[str] = field(default_factory=list)
    message_id: str = ""    

@dataclass
class EmailThread:
    thread_id: str
    messages: list[EmailMessage]