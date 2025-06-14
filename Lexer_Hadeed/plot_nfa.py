from graphviz import Digraph
import os
import re
from datetime import datetime

def plot_nfa_with_graphviz(nfa, regex_pattern="", out_dir="NFAs"):
    """Generate and save an NFA graph PNG under ./NFAs/, without inline display."""
    # Build Graphviz object
    dot = Digraph(format='png', engine='dot')
    dot.attr(
        rankdir='LR', dpi='300', size='20,10', ratio='compress',
        bgcolor='white', fontname='Arial', fontsize='14',
        nodesep='1.5', ranksep='2.0', splines='true', overlap='false'
    )
    dot.attr('node', fontname='Arial', fontsize='16', fontcolor='black',
             style='filled', fillcolor='lightblue', color='navy', penwidth='2')
    dot.attr('edge', fontname='Arial', fontsize='14', fontcolor='darkred',
             color='black', penwidth='2', arrowsize='1.2')

    # BFS to add all states & transitions
    visited = set()
    queue = [nfa.start]
    dot.node('START', style='invis', width='0')

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        # style start/accept
        if current == nfa.start:
            dot.node(str(current), style='filled,bold', fillcolor='lightgreen',
                     color='darkgreen', penwidth='3', xlabel='START')
        elif current == nfa.accept:
            dot.node(str(current), shape='doublecircle', style='filled,bold',
                     fillcolor='lightcoral', color='darkred', penwidth='3', xlabel='ACCEPT')
        else:
            dot.node(str(current))

        # transitions
        for symbol, targets in current.transitions.items():
            label = symbol.replace('"', '\\"').replace('\\', '\\\\')
            for target in targets:
                dot.edge(str(current), str(target), label=f' {label} ',
                         color='blue', fontcolor='blue', fontweight='bold')
                if target not in visited:
                    queue.append(target)

        # epsilon‐moves
        for eps in current.epsilon:
            dot.edge(str(current), str(eps), label=' ε ',
                     style='dashed', color='red', fontcolor='red', fontweight='bold')
            if eps not in visited:
                queue.append(eps)

    dot.edge('START', str(nfa.start), style='bold', color='green', penwidth='3')

    if regex_pattern:
        dot.attr(label=f'\\n\\nNFA for {regex_pattern}\\nLexer Token',
                 labelloc='t', fontsize='18', fontname='Arial Bold')

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Build safe filename
    if regex_pattern:
        base = re.sub(r'[^\w\-_\.]', '_', regex_pattern).strip('_')
        # Remove consecutive underscores and limit length
        base = re.sub(r'_+', '_', base)[:50]
    else:
        base = datetime.now().strftime("nfa_%Y%m%d_%H%M%S")
    
    # Ensure filename is not empty
    if not base:
        base = "nfa_unnamed"

    filename = base
    
     # Save directly to the NFAs folder
    full_path = os.path.join(out_dir, filename)
    dot.render(full_path, format='png', cleanup=True)

    out_path = f"{full_path}.png"
    print(f"✅ NFA graph saved to: {out_path}")
    
    return out_path