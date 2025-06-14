class State:
    _id_counter = 0

    def __init__(self):
        self.id = State._id_counter
        State._id_counter += 1
        self.transitions = {}  # symbol → set of states
        self.epsilon = set()

    def __str__(self):
        return f"q{self.id}"

    def __repr__(self):
        return self.__str__()

class NFA:
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept

def symbol_nfa(symbol):
    """Create NFA for a single symbol"""
    start = State()
    accept = State()
    start.transitions[symbol] = {accept}
    return NFA(start, accept)

def concat_nfa(nfa1, nfa2):
    """Concatenate two NFAs"""
    nfa1.accept.epsilon.add(nfa2.start)
    return NFA(nfa1.start, nfa2.accept)

def union_nfa(nfa1, nfa2):
    """Union of two NFAs"""
    start = State()
    accept = State()
    start.epsilon.update([nfa1.start, nfa2.start])
    nfa1.accept.epsilon.add(accept)
    nfa2.accept.epsilon.add(accept)
    return NFA(start, accept)

def star_nfa(nfa):
    """Kleene star of an NFA"""
    start = State()
    accept = State()
    start.epsilon.update([nfa.start, accept])
    nfa.accept.epsilon.update([nfa.start, accept])
    return NFA(start, accept)

def plus_nfa(nfa):
    """One or more repetitions of an NFA"""
    # Create a copy of the original NFA
    original = nfa
    star = star_nfa(nfa)
    return concat_nfa(original, star)

def print_nfa(nfa):
    """Print NFA structure"""
    visited = set()
    queue = [nfa.start]
    print(f"\nStart: {nfa.start}, Accept: {nfa.accept}\nTransitions:")

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        for symbol, states in current.transitions.items():
            for state in states:
                print(f"  {current} --{symbol}--> {state}")
                if state not in visited:
                    queue.append(state)

        for eps in current.epsilon:
            print(f"  {current} --ε--> {eps}")
            if eps not in visited:
                queue.append(eps)


def postfix_to_nfa(postfix_tokens):
    """Convert postfix tokens to NFA with enhanced word boundary support and error handling"""
    stack = []

    for token in postfix_tokens:
        if token == '*':
            if not stack:
                raise ValueError("Stack underflow: '*' operator needs one operand.")
            nfa = stack.pop()
            stack.append(star_nfa(nfa))

        elif token == '+':
            if not stack:
                raise ValueError("Stack underflow: '+' operator needs one operand.")
            nfa = stack.pop()
            stack.append(plus_nfa(nfa))

        elif token == '|':
            if len(stack) < 2:
                raise ValueError("Stack underflow: '|' operator needs two operands.")
            nfa2 = stack.pop()
            nfa1 = stack.pop()
            stack.append(union_nfa(nfa1, nfa2))

        elif token == '.':
            if len(stack) < 2:
                raise ValueError("Stack underflow: '.' operator needs two operands.")
            nfa2 = stack.pop()
            nfa1 = stack.pop()
            stack.append(concat_nfa(nfa1, nfa2))

        elif token == 'DIGIT':
            digit_nfa = symbol_nfa('0')
            for digit in '123456789':
                digit_nfa = union_nfa(digit_nfa, symbol_nfa(digit))
            stack.append(digit_nfa)

        elif token == 'SPACE':
            stack.append(symbol_nfa(' '))

        elif token == 'TAB':
            stack.append(symbol_nfa('\t'))

        elif token == 'NEWLINE':
            stack.append(symbol_nfa('\n'))

        elif token == 'WORD_BOUNDARY':
            # Skip actual \b handling here; used in lexer context
            continue

        elif token == 'WHITESPACE':
            space_char = union_nfa(symbol_nfa(' '), symbol_nfa('\t'))
            space_plus = concat_nfa(space_char, star_nfa(space_char))
            stack.append(space_plus)

        elif token == 'TERMINATOR':
            stack.append(symbol_nfa('\n'))

        elif token == 'NUMBER':
            digit_nfa = symbol_nfa('0')
            for digit in '123456789':
                digit_nfa = union_nfa(digit_nfa, symbol_nfa(digit))
            number_nfa = concat_nfa(digit_nfa, star_nfa(digit_nfa))
            stack.append(number_nfa)

        elif token == 'FLOAT':
            digit_nfa = symbol_nfa('0')
            for digit in '123456789':
                digit_nfa = union_nfa(digit_nfa, symbol_nfa(digit))
            digit_plus = concat_nfa(digit_nfa, star_nfa(digit_nfa))
            dot = symbol_nfa('.')
            float_nfa = concat_nfa(digit_plus, dot)
            float_nfa = concat_nfa(float_nfa, digit_plus)
            stack.append(float_nfa)

        elif token.startswith('[') and token.endswith(']'):
            char_class = token[1:-1]
            class_nfa = None
            i = 0
            while i < len(char_class):
                if i + 2 < len(char_class) and char_class[i + 1] == '-':
                    start_char = ord(char_class[i])
                    end_char = ord(char_class[i + 2])
                    range_nfa = symbol_nfa(chr(start_char))
                    for code in range(start_char + 1, end_char + 1):
                        range_nfa = union_nfa(range_nfa, symbol_nfa(chr(code)))
                    class_nfa = range_nfa if class_nfa is None else union_nfa(class_nfa, range_nfa)
                    i += 3
                else:
                    char_nfa = symbol_nfa(char_class[i])
                    class_nfa = char_nfa if class_nfa is None else union_nfa(class_nfa, char_nfa)
                    i += 1
            if class_nfa:
                stack.append(class_nfa)

        elif token.startswith('"') and token.endswith('"'):
            string_content = token[1:-1]
            if not string_content:
                empty_start = State()
                empty_accept = State()
                empty_start.epsilon.add(empty_accept)
                stack.append(NFA(empty_start, empty_accept))
            else:
                string_nfa = symbol_nfa(string_content[0])
                for char in string_content[1:]:
                    string_nfa = concat_nfa(string_nfa, symbol_nfa(char))
                stack.append(string_nfa)

        elif token.startswith('(') and token.endswith(')'):
            group_content = token[1:-1]
            if '|' in group_content:
                words = [w.strip().replace('\\b', '') for w in group_content.split('|')]
                group_nfa = None
                for word in words:
                    if not word:
                        continue
                    word_nfa = symbol_nfa(word[0])
                    for ch in word[1:]:
                        word_nfa = concat_nfa(word_nfa, symbol_nfa(ch))
                    group_nfa = word_nfa if group_nfa is None else union_nfa(group_nfa, word_nfa)
                if group_nfa:
                    stack.append(group_nfa)
            else:
                # Remove \b and build NFA
                word = group_content.replace('\\b', '')
                if word:
                    word_nfa = symbol_nfa(word[0])
                    for ch in word[1:]:
                        word_nfa = concat_nfa(word_nfa, symbol_nfa(ch))
                    stack.append(word_nfa)

        else:
            # Regular single character
            stack.append(symbol_nfa(token))

    if len(stack) != 1:
        raise ValueError(f"Invalid postfix expression: leftover items in stack: {stack}")

    return stack[0]

def combine_nfas(nfa_results):
    """Combine all NFAs into a single NFA with a new start state"""
    if not nfa_results:
        return None
        
    combined_start = State()
    accept_states = []

    for token_name, regex, nfa, _ in nfa_results:
        if nfa:  # Make sure NFA exists
            combined_start.epsilon.add(nfa.start)
            accept_states.append((nfa.accept, token_name))

    # Return NFA with list of accept states
    return NFA(combined_start, accept_states) 