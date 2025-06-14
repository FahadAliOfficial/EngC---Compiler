# dfa_table.py
"""
DFA Table Module
Converts DFA to transition table format for efficient lexical scanning.
"""

class DFATable:
    """Converts DFA to transition table for efficient scanning"""
    
    def __init__(self, start_state, all_states):
        self.start_state = start_state
        self.all_states = all_states
        self.alphabet = self._get_alphabet()
        self.state_map = {state: i for i, state in enumerate(all_states)}
        self.symbol_to_idx = {symbol: i for i, symbol in enumerate(self.alphabet)}
        self.transition_table = self._build_transition_table()
        self.accepting_info = self._build_accepting_info()
    
    def _get_alphabet(self):
        """Get sorted alphabet for consistent indexing"""
        alphabet = set()
        for state in self.all_states:
            alphabet.update(state.transitions.keys())
        return sorted(alphabet)
    
    def _build_transition_table(self):
        """Build 2D transition table: [state_idx][symbol] -> state_idx"""
        print("🔧 Building transition table...")
        
        # Initialize table with -1 (no transition)
        table = [[-1 for _ in self.alphabet] for _ in self.all_states]
        
        # Fill transition table
        for state in self.all_states:
            state_idx = self.state_map[state]
            for symbol, target in state.transitions.items():
                symbol_idx = self.symbol_to_idx[symbol]
                target_idx = self.state_map[target]
                table[state_idx][symbol_idx] = target_idx
        
        return table
    
    def _build_accepting_info(self):
        """Build accepting state information"""
        accepting_info = {}
        for i, state in enumerate(self.all_states):
            if state.is_accepting:
                accepting_info[i] = state.token_type
        return accepting_info
    
    def get_transition(self, state_idx, symbol):
        """Get transition from state via symbol"""
        if symbol not in self.symbol_to_idx:
            return -1
        symbol_idx = self.symbol_to_idx[symbol]
        return self.transition_table[state_idx][symbol_idx]
    
    def is_accepting(self, state_idx):
        """Check if state is accepting"""
        return state_idx in self.accepting_info
    
    def get_token_type(self, state_idx):
        """Get token type for accepting state"""
        return self.accepting_info.get(state_idx)
    
    def get_start_state_idx(self):
        """Get index of start state"""
        return self.state_map[self.start_state]
    
    def print_table_stats(self):
        """Print table statistics"""
        total_transitions = sum(1 for row in self.transition_table for cell in row if cell != -1)
        print(f"📊 Transition Table Stats:")
        print(f"   States: {len(self.all_states)}")
        print(f"   Alphabet: {len(self.alphabet)} symbols")
        print(f"   Transitions: {total_transitions}")
        print(f"   Accepting states: {len(self.accepting_info)}")
        print(f"   Table size: {len(self.all_states)} × {len(self.alphabet)} = {len(self.all_states) * len(self.alphabet)} cells")
    
    def export_table(self):
        """Export table data for external use"""
        return {
            'transition_table': self.transition_table,
            'accepting_info': self.accepting_info,
            'alphabet': self.alphabet,
            'state_count': len(self.all_states),
            'start_state_idx': self.get_start_state_idx(),
            'symbol_to_idx': self.symbol_to_idx
        }
    
    def print_table(self):
        """Print the complete transition table for debugging"""
        print(f"\n📋 Complete Transition Table ({len(self.all_states)} states, {len(self.alphabet)} symbols):")
        
        # Header
        header = "State".ljust(8) + "".join(f"{sym}".ljust(6) for sym in self.alphabet)
        print(header)
        print("-" * len(header))
        
        # Rows - shows ALL states and ALL symbols
        for i, state in enumerate(self.all_states):
            row = f"D{i}".ljust(8)
            for j, symbol in enumerate(self.alphabet):
                target_idx = self.transition_table[i][j]
                if target_idx == -1:
                    row += "∅".ljust(6)
                else:
                    row += f"D{target_idx}".ljust(6)
            
            # Mark accepting states
            if self.is_accepting(i):
                row += f" [{self.get_token_type(i)}]"
            
            print(row)

    def export_to_excel(self, filename: str = "dfa.xlsx"):
        """
        Export the DFA transition table and accepting states to an Excel file.
        - Sheet "Transitions" has the full transition table (with no-edges blank).
        - Sheet "AcceptingStates" has a list of accepting-state → token_type.
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("Please install pandas (and openpyxl) to use export_to_excel")

        # Build DataFrame for transition table
        df = pd.DataFrame(
            self.transition_table,
            index=[f"D{i}" for i in range(len(self.all_states))],
            columns=self.alphabet
        )

        # Replace all “no-transition” markers (-1) with blanks for readability
        df.replace(-1, "", inplace=True)

        # Build DataFrame for accepting states
        acc_df = pd.DataFrame([
            {"State": f"D{i}", "TokenType": tok}
            for i, tok in self.accepting_info.items()
        ])

        # Write to Excel
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Transitions")
            acc_df.to_excel(writer, sheet_name="AcceptingStates", index=False)

        print(f"✅ DFA exported to Excel file: {filename}")
