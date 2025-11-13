class ContextItem:
    NEXT_ID = 1

    def __init__(self, content: str):
        self.id = ContextItem.NEXT_ID
        ContextItem.NEXT_ID += 1
        self.content = content

    def __str__(self):
        return f"CONTEXT #{self.id}: {self.content}"

class Notification(ContextItem):
    def __init__(self, content: str):
        super().__init__(content)

    def __str__(self):
        return f"NOTIFICATION #{self.id}: {self.content}"


class AgentContext:
    def __init__(self):
        self.context_items: dict[int, ContextItem] = {}
        self.notifications: dict[int, Notification] = {}

    def add_context(self, content: str):
        context_item = ContextItem(content)
        self.context_items[context_item.id] = context_item

    def remove_context(self, context_id: int):
        if context_id in self.context_items:
            del self.context_items[context_id]

    def get_context(self) -> str:
        return "\n".join(str(item) for item in self.context_items.values())
    
    def clear_context(self):
        self.context_items = {}

    def add_notification(self, content: str):
        notification = Notification(content)
        self.notifications[notification.id] = notification

    def remove_notification(self, notification_id: int):
        if notification_id in self.notifications:
            del self.notifications[notification_id]

    def get_notifications(self) -> str:
        return "\n".join(str(note) for note in self.notifications.values())
    
    def clear_notifications(self):
        self.notifications = {}