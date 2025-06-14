def tokenize_regex(regex):
    """Enhanced tokenizer that handles regex patterns properly with word boundaries"""
    tokens = []
    i = 0
    
    # Check if this is a word-based pattern (contains |)
    if '|' in regex and not regex.startswith(r'\d') and not regex.startswith('['):
        # This is a word alternation pattern like "if|else|for"
        # We need to add word boundaries to ensure exact matching
        words = regex.split('|')
        word_nfas = []
        
        for word in words:
            word = word.strip()
            if word:
                # Create exact word match with word boundaries
                tokens.append(f'\\b{word}\\b')
        
        # Join with alternation
        return ['(' + '|'.join(f'\\b{word.strip()}\\b' for word in words if word.strip()) + ')']
    
    # Handle regular regex patterns
    while i < len(regex):
        char = regex[i]
        
        # Handle escape sequences
        if char == '\\' and i + 1 < len(regex):
            next_char = regex[i + 1]
            if next_char == 'd':
                tokens.append('DIGIT')  # \d becomes DIGIT token
            elif next_char == 's':
                tokens.append('SPACE')  # \s becomes SPACE token
            elif next_char == 'n':
                tokens.append('NEWLINE')  # \n becomes NEWLINE token
            elif next_char == 't':
                tokens.append('TAB')    # \t becomes TAB token
            elif next_char == 'b':
                tokens.append('WORD_BOUNDARY')  # \b becomes WORD_BOUNDARY token
            elif next_char in '(){}[].*+?|^$\\':
                # For escaped special chars, add just the character
                tokens.append(next_char)
            else:
                # For other escaped characters, keep both
                tokens.append(regex[i:i+2])
            i += 2
            continue
        
        # Handle character classes [...]
        elif char == '[':
            bracket_content = ''
            i += 1
            while i < len(regex) and regex[i] != ']':
                bracket_content += regex[i]
                i += 1
            if i < len(regex):  # consume closing ]
                i += 1
            tokens.append(f'[{bracket_content}]')
            continue
        
        # Handle quoted strings
        elif char == '"':
            quote_content = char
            i += 1
            while i < len(regex) and regex[i] != '"':
                if regex[i] == '\\' and i + 1 < len(regex):
                    # Handle escaped characters in strings
                    quote_content += regex[i:i+2]
                    i += 2
                else:
                    quote_content += regex[i]
                    i += 1
            if i < len(regex):  # consume closing "
                quote_content += regex[i]
                i += 1
            tokens.append(quote_content)
            continue
            
        # Handle parentheses groups
        elif char == '(':
            paren_content = ''
            paren_count = 1
            i += 1
            while i < len(regex) and paren_count > 0:
                if regex[i] == '(':
                    paren_count += 1
                elif regex[i] == ')':
                    paren_count -= 1
                if paren_count > 0:
                    paren_content += regex[i]
                i += 1
            tokens.append(f'({paren_content})')
            continue
        
        # Regular characters and operators
        else:
            tokens.append(char)
            i += 1
    
    return tokens