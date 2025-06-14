class DFAState:
    _id_counter = 0

    def __init__(self, nfa_states):
        self.id = DFAState._id_counter
        DFAState._id_counter += 1
        self.nfa_states = frozenset(nfa_states)
        self.transitions = {}  # symbol → DFAState
        self.is_accepting = False
        self.token_type = None

    def __str__(self):
        return f"D{self.id}"

    def __repr__(self):
        return self.__str__()

def epsilon_closure(states):
    """Compute epsilon closure of a set of NFA states - OPTIMIZED"""
    closure = set(states)
    stack = list(states)

    while stack:
        state = stack.pop()
        for eps in state.epsilon:
            if eps not in closure:
                closure.add(eps)
                stack.append(eps)
    return closure

def move(states, symbol):
    """Move from a set of NFA states via a symbol"""
    result = set()
    for state in states:
        if symbol in state.transitions:
            result.update(state.transitions[symbol])
    return result

def get_alphabet(nfa):
    """Collect all symbols used in the NFA transitions (excluding epsilon)"""
    visited = set()
    alphabet = set()
    stack = [nfa.start]

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        for symbol, targets in current.transitions.items():
            alphabet.add(symbol)
            for target in targets:
                if target not in visited:
                    stack.append(target)

        for eps in current.epsilon:
            if eps not in visited:
                stack.append(eps)
    
    return alphabet

def nfa_to_dfa(nfa):
    """Convert NFA to DFA - Complete conversion without artificial limits"""
    alphabet = get_alphabet(nfa)
    print(f"Alphabet size: {len(alphabet)}")
    
    dfa_states = {}
    dfa_start_closure = epsilon_closure({nfa.start})
    start_dfa = DFAState(dfa_start_closure)

    # Handle accept states - nfa.accept should be a list of (state, token_name) tuples
    accept_dict = {}
    if isinstance(nfa.accept, list):
        for state, token in nfa.accept:
            accept_dict[state] = token
    else:
        accept_dict[nfa.accept] = None

    # Check if start state is accepting
    for state in start_dfa.nfa_states:
        if state in accept_dict:
            start_dfa.is_accepting = True
            start_dfa.token_type = accept_dict[state]
            break

    dfa_states[start_dfa.nfa_states] = start_dfa
    unmarked = [start_dfa]
    all_dfa_states = [start_dfa]

    # Process all states without limits
    processed = 0
    total_transitions = 0

    while unmarked:
        current_dfa = unmarked.pop(0)
        processed += 1
        
        # Progress reporting for large conversions
        if processed % 100 == 0:
            print(f"Processed {processed} DFA states...")
        
        for symbol in alphabet:
            move_result = move(current_dfa.nfa_states, symbol)
            if not move_result:
                continue
            
            closure = epsilon_closure(move_result)
            closure_frozen = frozenset(closure)

            if closure_frozen not in dfa_states:
                new_dfa = DFAState(closure)
                # Check if any state in closure is accepting
                for state in closure:
                    if state in accept_dict:
                        new_dfa.is_accepting = True
                        new_dfa.token_type = accept_dict[state]
                        break
                        
                dfa_states[closure_frozen] = new_dfa
                unmarked.append(new_dfa)
                all_dfa_states.append(new_dfa)

            current_dfa.transitions[symbol] = dfa_states[closure_frozen]
            total_transitions += 1

    # Print comprehensive statistics
    print(f"\n=== DFA Conversion Statistics ===")
    print(f"Total DFA states: {len(all_dfa_states)}")
    print(f"Total transitions (edges): {total_transitions}")
    print(f"Alphabet size: {len(alphabet)}")
    
    # Count accepting states
    accepting_states = sum(1 for state in all_dfa_states if state.is_accepting)
    print(f"Accepting states: {accepting_states}")
    
    # Calculate transition density
    max_possible_transitions = len(all_dfa_states) * len(alphabet)
    if max_possible_transitions > 0:
        density = (total_transitions / max_possible_transitions) * 100
        print(f"Transition density: {density:.2f}%")
    
    print(f"================================\n")

    return start_dfa, all_dfa_states