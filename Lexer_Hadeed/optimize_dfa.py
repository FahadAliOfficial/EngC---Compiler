# dfa_optimizer.py
"""
DFA Optimization Module
Provides functionality to optimize DFA by removing unreachable states and minimizing using partition refinement.
"""

from collections import defaultdict
from copy import deepcopy

class DFAOptimizer:
    """Optimizes DFA by removing unreachable states and merging equivalent states"""
    
    def __init__(self, start_dfa, all_states):
        self.start_dfa = start_dfa
        self.all_states = all_states
        self.alphabet = self._get_alphabet()
    
    def _get_alphabet(self):
        """Extract alphabet from DFA transitions"""
        alphabet = set()
        for state in self.all_states:
            alphabet.update(state.transitions.keys())
        return sorted(alphabet)
    
    def remove_unreachable_states(self):
        """Remove states that are unreachable from start state"""
        print("🔧 Removing unreachable states...")
        
        # BFS to find reachable states
        reachable = set()
        queue = [self.start_dfa]
        reachable.add(self.start_dfa)
        
        while queue:
            current = queue.pop(0)
            for symbol, target in current.transitions.items():
                if target not in reachable:
                    reachable.add(target)
                    queue.append(target)
        
        # Filter out unreachable states
        original_count = len(self.all_states)
        self.all_states = [state for state in self.all_states if state in reachable]
        removed = original_count - len(self.all_states)
        
        print(f"   Removed {removed} unreachable states ({len(self.all_states)} remaining)")
        return self.all_states
    
    def minimize_dfa(self):
        """Minimize DFA using partition refinement algorithm"""
        print("🔧 Minimizing DFA...")
        
        # Step 1: Remove unreachable states
        reachable_states = self.remove_unreachable_states()
        
        # Step 2: Initial partition (accepting vs non-accepting)
        accepting_states = [s for s in reachable_states if s.is_accepting]
        non_accepting_states = [s for s in reachable_states if not s.is_accepting]
        
        # Further partition accepting states by token type
        accepting_by_token = defaultdict(list)
        for state in accepting_states:
            accepting_by_token[state.token_type].append(state)
        
        # Create initial partitions
        partitions = []
        if non_accepting_states:
            partitions.append(non_accepting_states)
        for token_states in accepting_by_token.values():
            partitions.append(token_states)
        
        print(f"   Initial partitions: {len(partitions)}")
        
        # Step 3: Refine partitions
        changed = True
        iteration = 0
        
        while changed and iteration < 100:  # Safety limit
            changed = False
            iteration += 1
            new_partitions = []
            
            for partition in partitions:
                # Try to split this partition
                sub_partitions = self._split_partition(partition, partitions)
                if len(sub_partitions) > 1:
                    changed = True
                new_partitions.extend(sub_partitions)
            
            partitions = new_partitions
            print(f"   Iteration {iteration}: {len(partitions)} partitions")
        
        # Step 4: Build minimized DFA
        return self._build_minimized_dfa(partitions)
    
    def _split_partition(self, partition, all_partitions):
        """Split a partition based on transition behavior"""
        if len(partition) <= 1:
            return [partition]
        
        # Group states by their transition signatures
        signature_groups = defaultdict(list)
        
        for state in partition:
            signature = []
            for symbol in self.alphabet:
                if symbol in state.transitions:
                    target = state.transitions[symbol]
                    # Find which partition the target belongs to
                    target_partition_idx = None
                    for i, part in enumerate(all_partitions):
                        if target in part:
                            target_partition_idx = i
                            break
                    signature.append(target_partition_idx)
                else:
                    signature.append(None)
            
            signature_groups[tuple(signature)].append(state)
        
        return list(signature_groups.values())
    
    def _build_minimized_dfa(self, partitions):
        """Build new minimized DFA from partitions"""
        print("🔧 Building minimized DFA...")
        
        # Create representative states
        partition_reps = {}  # partition_idx -> new_state
        state_to_partition = {}  # old_state -> partition_idx
        
        # Map old states to partition indices
        for i, partition in enumerate(partitions):
            for state in partition:
                state_to_partition[state] = i
        
        # Create new states (one per partition)
        new_states = []
        for i, partition in enumerate(partitions):
            # Use first state in partition as representative
            rep_state = partition[0]
            
            # Create new state with same properties
            new_state = deepcopy(rep_state)
            new_state.id = i
            new_state.transitions = {}
            
            partition_reps[i] = new_state
            new_states.append(new_state)
        
        # Build transitions for new states
        for i, partition in enumerate(partitions):
            rep_state = partition[0]  # Representative of this partition
            new_state = partition_reps[i]
            
            for symbol in self.alphabet:
                if symbol in rep_state.transitions:
                    target = rep_state.transitions[symbol]
                    target_partition = state_to_partition[target]
                    new_state.transitions[symbol] = partition_reps[target_partition]
        
        # Find new start state
        start_partition = state_to_partition[self.start_dfa]
        new_start = partition_reps[start_partition]
        
        print(f"✅ Minimization complete: {len(self.all_states)} → {len(new_states)} states")
        return new_start, new_states
    
    def optimize(self):
        """Main optimization method - convenience function"""
        return self.minimize_dfa()

