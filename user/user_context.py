class ContextItem:
    NEXT_ID = 1

    def __init__(self, content: str):
        self.id = ContextItem.NEXT_ID
        ContextItem.NEXT_ID += 1
        self.content = content


class DescriptiveStatement:
    NEXT_ID = 1

    def __init__(self, content: str, confidence: float):
        self.id = DescriptiveStatement.NEXT_ID
        DescriptiveStatement.NEXT_ID += 1
        self.content = content
        self.confidence = confidence


class UserContext:
    def __init__(self):
        self.context_items: list[ContextItem] = []
        self.descriptive_statements: list[DescriptiveStatement] = []

    def add_context(self, content: str):
        context_item = ContextItem(content)
        self.context_items.append(context_item)

    def add_descriptive_statement(self, content: str, confidence: float):
        statement = DescriptiveStatement(content, confidence)
        self.descriptive_statements.append(statement)

    def get_context(self) -> str:
        return "\n".join(item.content for item in self.context_items)