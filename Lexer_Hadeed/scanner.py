# lexical_scanner.py
"""
Lexical Scanner Module
Table-driven lexical scanner for tokenizing input text.
"""

import re

class Token:
    def __init__(self, name, lexeme):
        self.name = name
        self.lexeme = lexeme

    def __str__(self):
        return f"{self.name}({self.lexeme})"

    def __repr__(self):
        return str(self)

class TokenStream:
    """Manages a stream of tokens for parsing"""
    
    def __init__(self, tokens):
        self.tokens = tokens
        self.position = 0
    
    def current(self):
        """Get current token without advancing"""
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None
    
    def peek(self, offset=1):
        """Peek at token at current position + offset"""
        pos = self.position + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None
    
    def advance(self):
        """Move to next token and return it"""
        if self.position < len(self.tokens):
            token = self.tokens[self.position]
            self.position += 1
            return token
        return None
    
    def has_more(self):
        """Check if there are more tokens"""
        return self.position < len(self.tokens)
    
    def expect(self, token_type):
        """Advance and expect a specific token type"""
        token = self.advance()
        if token is None or token.name != token_type:
            raise SyntaxError(f"Expected {token_type}, got {token.name if token else 'EOF'}")
        return token
    
    def match(self, *token_types):
        """Check if current token matches any of the given types"""
        current = self.current()
        if current is None:
            return False
        return current.name in token_types
    
    def consume_if(self, token_type):
        """Consume token if it matches the type, otherwise return None"""
        if self.match(token_type):
            return self.advance()
        return None
    
    def reset(self):
        """Reset position to beginning"""
        self.position = 0
    
    def __len__(self):
        return len(self.tokens)
    
    def __iter__(self):
        return iter(self.tokens)

class ScannerError(Exception):
    """Exception raised during lexical scanning"""
    
    def __init__(self, message, line_number=1, position=0, character=None):
        self.message = message
        self.line_number = line_number
        self.position = position
        self.character = character
        super().__init__(self.format_message())
    
    def format_message(self):
        """Format error message with location info"""
        location = f"line {self.line_number}, position {self.position}"
        if self.character:
            return f"{self.message} at {location}: '{self.character}'"
        return f"{self.message} at {location}"
    

class LexicalScanner:
    
    # near the top of your scanner class, define:
    KEYWORDS = {'if','else','for','while','main','do'}
    TYPES    = {'integer','float','string','boolean'}
    BOOLS    = {'true','false'}
    ARITH_OP = {'add','subtract','multiply','divide','remainder','power'}
    LOGIC_OP = {'and','or','not'}
    
    FLOAT_RE = re.compile(r'\d+\.\d+')
    STRING_RE = re.compile(r'"[^"\n]*"')

    """Table-driven lexical scanner with enhanced filtering"""
    
    def __init__(self, dfa_table, skip_whitespace=True, skip_terminators=True, skip_comments=True):
        self.table = dfa_table
        self.start_state_idx = self.table.get_start_state_idx()
        self.skip_whitespace = skip_whitespace
        self.skip_terminators = skip_terminators  # New option
        self.skip_comments = skip_comments
        self.error_recovery = True
    
    def _should_include_token(self, token):
        """Determine if token should be included in output - ENHANCED"""
        # Skip whitespace if configured
        if self.skip_whitespace and token.name in {'WHITESPACE', 'SPACE', 'TAB'}:
            return False
        
        # Skip terminators (newlines) if configured - NEW
        if self.skip_terminators and token.name in {'TERMINATOR', 'NEWLINE'}:
            return False
        
        # Skip comments if configured
        if self.skip_comments and token.name in {'COMMENT', 'SINGLE_LINE_COMMENT', 'MULTI_LINE_COMMENT'}:
            return False
        
        return True
    
    # [Rest of the scanner methods remain the same...]
    

    def scan(self, input_text):
        print(f"🔍 Scanning input: {len(input_text)} characters")
        tokens = []
        pos = 0
        line_number = 1

        while pos < len(input_text):
            # 1) FLOATs
            m = self.FLOAT_RE.match(input_text, pos)
            if m:
                tokens.append(Token('FLOAT', m.group(0)))
                pos += len(m.group(0))
                continue

            # 2) STRINGs
            m = self.STRING_RE.match(input_text, pos)
            if m:
                tokens.append(Token('STRING', m.group(0)))
                pos += len(m.group(0))
                continue


            # 4) Your existing DFA fallback
            token, new_pos = self._scan_token(input_text, pos, line_number)
            if token is None:
                if self.error_recovery:
                    char = input_text[pos]
                    print(f"❌ Lexical error at line {line_number}, position {pos}: unexpected character '{char}'")
                    pos += 1
                    continue
                else:
                    raise ScannerError("Unexpected character", line_number, pos, input_text[pos])

            # update line_number, filters, etc., as before…
            if '\n' in token.lexeme:
                line_number += token.lexeme.count('\n')
            if self._should_include_token(token):
                tokens.append(token)
            pos = new_pos

        tokens.append(Token('$','$'))
        print(f"✅ Scanning complete: {len(tokens)} tokens generated")
        return tokens
    
    def _scan_token(self, text, start_pos, line_number):
        """Scan a single token with enhanced word boundary checking"""
        current_state = self.start_state_idx
        position = start_pos
        last_accepting_pos = -1
        last_accepting_token = None

        # Simulate DFA
        while position < len(text):
            char = text[position]
            next_state = self.table.get_transition(current_state, char)

            if next_state == -1:
                break

            current_state = next_state
            position += 1

            if self.table.is_accepting(current_state):
                lexeme = text[start_pos:position]
                token_type = self.table.get_token_type(current_state)

                # Enhanced word boundary checking for keywords and operators
                if self._is_word_token(token_type):
                    if self._check_word_boundaries(text, start_pos, position):
                        last_accepting_pos = position
                        last_accepting_token = Token(token_type, lexeme)
                else:
                    last_accepting_pos = position
                    last_accepting_token = Token(token_type, lexeme)

        if last_accepting_token is not None:
            lex = last_accepting_token.lexeme
            # --- FLOAT/NUMBER sanity override (two‐way) ---
            if last_accepting_token.name == 'FLOAT' and '.' not in lex:
                # our DFA sometimes tags “10” as FLOAT → force back to NUMBER
                last_accepting_token.name = 'NUMBER'
            elif last_accepting_token.name == 'NUMBER' and '.' in lex:
                # if we saw “123.45” but got NUMBER → promote to FLOAT
                last_accepting_token.name = 'FLOAT'
            # --------------------------------------------------

            # --- Step 2: post-match override for keywords & ops vs. identifiers ---
            if last_accepting_token.name == 'IDENTIFIER':
                lex = last_accepting_token.lexeme
                if lex in self.KEYWORDS:
                    last_accepting_token.name = 'KEYWORD'
                elif lex in self.TYPES:
                    last_accepting_token.name = 'TYPE'
                elif lex in self.BOOLS:
                    last_accepting_token.name = 'BOOL'
                elif lex in self.ARITH_OP:
                    last_accepting_token.name = 'ARITHMETIC_OP'
                elif lex in self.LOGIC_OP:
                    last_accepting_token.name = 'LOGICAL_OP'
            # -----------------------------------------------------------------------

            return last_accepting_token, last_accepting_pos
        else:
            return None, start_pos + 1

    
    def _is_word_token(self, token_type):
        """Check if token type requires word boundary validation"""
        return token_type in {'KEYWORD', 'TYPE', 'BOOL', 'ARITHMETIC_OP', 'LOGICAL_OP', 
                             'COMMA', 'DOT', 'COLON', 'SEMI'}
    
    def _check_word_boundaries(self, text, start_pos, end_pos):
        """Check if token has proper word boundaries"""
        # Check character before token
        if start_pos > 0:
            prev_char = text[start_pos - 1]
            if prev_char.isalnum() or prev_char == '_':
                return False
        
        # Check character after token
        if end_pos < len(text):
            next_char = text[end_pos]
            if next_char.isalnum() or next_char == '_':
                return False
        
        return True
    
    def scan_to_stream(self, input_text):
        """Scan input and return a TokenStream"""
        tokens = self.scan(input_text)
        return TokenStream(tokens)
    
    def validate_input(self, input_text):
        """
        Validate input without generating tokens
        
        Returns:
            tuple: (is_valid, error_messages)
        """
        errors = []
        position = 0
        line_number = 1
        
        # Temporarily disable error recovery
        original_recovery = self.error_recovery
        self.error_recovery = False
        
        try:
            while position < len(input_text):
                try:
                    token, new_position = self._scan_token(input_text, position, line_number)
                    if token is None:
                        char = input_text[position]
                        errors.append(f"Line {line_number}: Unexpected character '{char}' at position {position}")
                        position += 1
                    else:
                        if token.lexeme == '\n' or '\n' in token.lexeme:
                            line_number += token.lexeme.count('\n')
                        position = new_position
                except Exception as e:
                    errors.append(f"Line {line_number}: {str(e)}")
                    position += 1
        
        finally:
            self.error_recovery = original_recovery
        
        return len(errors) == 0, errors
    
    def get_scanner_stats(self):
        """Get scanner statistics"""
        return {
            'states': len(self.table.all_states),
            'alphabet_size': len(self.table.alphabet),
            'transitions': sum(1 for row in self.table.transition_table for cell in row if cell != -1),
            'accepting_states': len(self.table.accepting_info),
            'skip_whitespace': self.skip_whitespace,
            'skip_comments': self.skip_comments,
            'error_recovery': self.error_recovery
        }
    
    def print_scanner_info(self):
        """Print scanner information"""
        stats = self.get_scanner_stats()
        print("🔍 Lexical Scanner Information:")
        print(f"   States: {stats['states']}")
        print(f"   Alphabet size: {stats['alphabet_size']}")
        print(f"   Transitions: {stats['transitions']}")
        print(f"   Accepting states: {stats['accepting_states']}")
        print(f"   Skip whitespace: {stats['skip_whitespace']}")
        print(f"   Skip comments: {stats['skip_comments']}")
        print(f"   Error recovery: {stats['error_recovery']}")

def create_scanner(dfa_table, **options):
    """
    Convenience function to create a lexical scanner
    
    Args:
        dfa_table: DFATable object
        **options: Scanner options (skip_whitespace, skip_comments, etc.)
    
    Returns:
        LexicalScanner: Configured scanner
    """
    scanner = LexicalScanner(dfa_table, **options)
    scanner.print_scanner_info()
    return scanner

# Test and utility functions
def test_scanner(scanner, test_input):
    """Test scanner with sample input"""
    print(f"\n🧪 Testing scanner:")
    print(f"Input: {repr(test_input)}")
    print(f"Length: {len(test_input)} characters")
    
    try:
        tokens = scanner.scan(test_input)
        
        print(f"\n📝 Generated tokens ({len(tokens)}):")
        for i, token in enumerate(tokens):
            print(f"  {i+1:2d}. {token}")
        
        return tokens
    
    except ScannerError as e:
        print(f"❌ Scanner error: {e}")
        return []

def analyze_tokens(tokens):
    """Analyze token distribution"""
    if not tokens:
        return
    
    from collections import Counter
    
    token_counts = Counter(token.name for token in tokens)
    
    print(f"\n📊 Token Analysis:")
    print(f"Total tokens: {len(tokens)}")
    print(f"Unique token types: {len(token_counts)}")
    print(f"\nToken distribution:")
    
    for token_type, count in token_counts.most_common():
        percentage = (count / len(tokens)) * 100
        print(f"  {token_type}: {count} ({percentage:.1f}%)")
