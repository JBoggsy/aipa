class UserContext:
    def __init__(self):
        self.context_items = []

    def add_context(self, context: str):
        self.context_items.append(context)

    def get_context(self) -> str:
        return "\n".join(self.context_items)