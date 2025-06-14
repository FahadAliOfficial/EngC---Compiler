class Token:
    def __init__(self, name, lexeme):
        self.name = name
        self.lexeme = lexeme

    def __str__(self):
        return f"{self.name}({self.lexeme})"

    def __repr__(self):
        return str(self)