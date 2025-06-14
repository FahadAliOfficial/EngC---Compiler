token_defs = [
    # Specific word-based tokens first (most specific to least specific)
    ("KEYWORD", "if|else|for|while|main|do"),
    ("TYPE", "integer|float|string|boolean"),
    ("BOOL", "true|false"),
    ("ARITHMETIC_OP", "add|subtract|multiply|divide|remainder|power"),
    ("RELATIONAL_OP", "is equal to|is not equal to|is greater than|is less than|is greater than or equal to|is less than or equal to"),
    ("LOGICAL_OP", "and|or|not"),
    ("ASSIGN_OP", "equals to|add equals to|subtract equals to|multiply equals to|divide equals to|remainder equals to"),
    ("COMMA", "comma"),
    ("DOT", "dot"),
    ("COLON", "colon"),
    ("SEMI", "semicolon"),
    
    # Number patterns (FLOAT must come before NUMBER for proper matching)
    ("FLOAT", r"\d+\.\d+"),     # More specific - matches digits.digits
    ("NUMBER", r"\d+"),         # More general - matches just digits
    
    # String patterns
    ("STRING",   r'"[^"\n]*"'), 
    
    # Generic identifier (MUST come after all specific word tokens)
    ("IDENTIFIER", "[a-zA-Z_][a-zA-Z_0-9]*"),
    
    # Punctuation
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    
    # Whitespace and terminators (FIXED)
    ("WHITESPACE", r"[ \t]+"),   # One or more spaces or tabs
    ("TERMINATOR", r"\n")        # Single newline
]
