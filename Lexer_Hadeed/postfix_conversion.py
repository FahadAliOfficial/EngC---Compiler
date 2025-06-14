def to_postfix(tokens):
    precedence = {'*': 3, '+': 3, '?': 3, '.': 2, '|': 1}
    output = []
    stack = []

    if len(tokens) == 1:
        return tokens
    if not tokens:
        return []

    # Add implicit concatenation
    processed_tokens = []
    for i in range(len(tokens)):
        curr = tokens[i]
        processed_tokens.append(curr)

        if i + 1 < len(tokens):
            next_ = tokens[i + 1]

            # If current is operand or closure and next is operand or open paren
            if (
                curr not in ['|', '.', '('] and
                next_ not in ['|', '.', ')', '*', '+', '?']
            ):
                processed_tokens.append('.')

    for token in processed_tokens:
        if token in precedence:
            while (
                stack and stack[-1] != '(' and
                precedence.get(stack[-1], 0) >= precedence[token]
            ):
                output.append(stack.pop())
            stack.append(token)

        elif token == '(':
            stack.append(token)

        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if stack and stack[-1] == '(':
                stack.pop()
        else:
            output.append(token)

    while stack:
        top = stack.pop()
        if top != '(':
            output.append(top)

    return output
