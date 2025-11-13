from base64 import urlsafe_b64decode


class EmailMessage:
    """
    Represents a single email message.
    """
    def __init__(
        self,
        subject: str,
        sender: str,
        recipients: list[str],
        body: str,
        timestamp: str,
        labels: list[str] | None = None,
        message_id: str = ""
    ):
        self.subject = subject
        self.sender = sender
        self.recipients = recipients
        self.body = body
        self.timestamp = timestamp
        self.labels = labels if labels is not None else []
        self.message_id = message_id

    @property
    def recipients_str(self) -> str:
        return ", ".join(self.recipients)

    def as_formatted_string(self) -> str:
        return (f"From: {self.sender}\n"
                f"To: {self.recipients_str}\n"
                f"Subject: {self.subject}\n"
                f"Date: {self.timestamp}\n\n"
                f"{self.body}")

    def __str__(self):
        return f"EmailMessage(subject={self.subject}, sender={self.sender}, recipients={self.recipients}, date={self.timestamp}, message_id={self.message_id})"


class EmailThread:
    """
    Represents a thread of email messages.
    """
    def __init__(
        self,
        thread_id: str,
        timestamp: str,
        messages: list[EmailMessage]
    ):
        self.thread_id = thread_id
        self.timestamp = timestamp
        self.messages = messages
        self.summary: str = ""

    def as_formatted_string(self) -> str:
        formatted_messages = "\n\n---\n\n".join(
            msg.as_formatted_string() for msg in self.messages
        )
        return f"Thread ID: {self.thread_id}\n\n{formatted_messages}"

    def __str__(self):
        return f"EmailThread(thread_id={self.thread_id}, messages_count={len(self.messages)})"