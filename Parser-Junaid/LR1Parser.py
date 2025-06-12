import copy
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def create_ast_png(ast_node, filename="parseTree.png"):
    """
    Create a beautiful PNG image of your AST with proper node spacing.
    
    Args:
        ast_node: The root AST node (must have .type, .value, .children)
        filename: Output PNG filename 
    
    Returns:
        str: Path to the created PNG file
    """
    if not ast_node:
        print("❌ No parse node provided")
        return None
        
    print(f"🎨 Creating Parse Tree Visualization in: {filename}")
    
    # Colors for different node types
    colors = {
        'Program': '#E74C3C',
        'MainFunction': '#3498DB', 
        'StatementList': '#2ECC71',
        'Declaration': '#F39C12',
        'Assignment': '#9B59B6',
        'IfStatement': '#E67E22',
        'WhileStatement': '#1ABC9C',
        'IDENTIFIER': '#FFB6C1',
        'NUMBER': '#FFA07A',
        'TYPE': '#98FB98',
        'KEYWORD': '#F0E68C',
        'default': '#BDC3C7'
    }
    
    # Calculate node width based on label size
    def get_node_width(node):
        label = node.type
        if hasattr(node, 'value') and node.value:
            label += f"\n{node.value}"
        return max(2.0, 0.5 * len(label))  # Minimum width of 2.0
    
    # First pass: Calculate subtree widths
    subtree_widths = {}
    
    def calc_subtree_width(node):
        if not hasattr(node, 'children') or not node.children:
            width = get_node_width(node)
        else:
            # Calculate total width needed for children with padding
            children_width = 0
            for child in node.children:
                child_width = calc_subtree_width(child)
                children_width += child_width + 1.0  # Add 1.0 as padding between siblings
            
            # Remove extra padding from last child
            if node.children:
                children_width -= 1.0
                
            # Node width is max of its own label width and children's total width
            node_width = get_node_width(node)
            width = max(node_width, children_width)
        
        subtree_widths[id(node)] = width
        return width
    
    # Calculate overall tree width
    total_width = calc_subtree_width(ast_node)
    
    # Second pass: Calculate actual positions
    positions = {}
    
    def calculate_positions(node, level, x_center):
        y_pos = -level * 3.5  # Increased vertical spacing
        positions[id(node)] = (x_center, y_pos)
        
        if hasattr(node, 'children') and node.children:
            # Calculate start position for first child
            child_count = len(node.children)
            
            if child_count == 1:
                # Single child is centered under parent
                calculate_positions(node.children[0], level + 1, x_center)
            else:
                # Calculate positions for multiple children
                total_child_width = sum(subtree_widths[id(child)] for child in node.children)
                total_padding = (child_count - 1) * 2.0  # Increased padding between siblings
                
                # Start position
                current_x = x_center - (total_child_width + total_padding) / 2
                
                for child in node.children:
                    child_width = subtree_widths[id(child)]
                    child_center = current_x + child_width / 2
                    calculate_positions(child, level + 1, child_center)
                    current_x += child_width + 2.0  # Move to next child position with padding
    
    # Start positioning from the center
    calculate_positions(ast_node, 0, total_width / 2)
    
    # Create figure with dynamic sizing based on tree width
    fig_width = max(20, total_width * 1.2)  # Minimum width of 20, but scale up for wide trees
    fig_height = 14
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
    
    # Determine good aspect ratio
    x_coords = [pos[0] for pos in positions.values()]
    y_coords = [pos[1] for pos in positions.values()]
    
    if x_coords and y_coords:
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords) 
        
        if y_range > 0 and x_range > 0:
            # Set aspect ratio that ensures enough vertical space
            aspect_ratio = 0.5 * x_range / y_range
            ax.set_aspect(aspect_ratio)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Set background
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')
    
    # Draw connections and nodes
    def draw_tree(node):
        pos = positions[id(node)]
        
        # Draw connections to children
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                child_pos = positions[id(child)]
                ax.plot([pos[0], child_pos[0]], [pos[1], child_pos[1]], 
                       'k-', linewidth=1.5, alpha=0.6, zorder=1)
        
        # Draw node
        label = node.type
        if hasattr(node, 'value') and node.value:
            label += f"\n{node.value}"
        
        color = colors.get(node.type, colors['default'])
        
        # Scale box size based on label length
        box_width = min(max(1.2, len(label) * 0.2), 4.0)
        box_height = 0.8 if '\n' in label else 0.5
        
        box = FancyBboxPatch(
            (pos[0] - box_width/2, pos[1] - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.3",
            facecolor=color,
            edgecolor='#2C3E50',
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(box)
        
        ax.text(pos[0], pos[1], label, 
               fontsize=9, fontweight='bold',
               ha='center', va='center',
               color='#2C3E50', zorder=3)
        
        # Draw children
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                draw_tree(child)
    
    draw_tree(ast_node)
    
    # Set limits with margins
    if positions:
        x_coords = [pos[0] for pos in positions.values()]
        y_coords = [pos[1] for pos in positions.values()]
        margin_x = (max(x_coords) - min(x_coords)) * 0.1 + 2.0
        margin_y = (max(y_coords) - min(y_coords)) * 0.1 + 1.0
        ax.set_xlim(min(x_coords) - margin_x, max(x_coords) + margin_x)
        ax.set_ylim(min(y_coords) - margin_y, max(y_coords) + margin_y)
    
    # Add title
    plt.title("Parse Tree", 
             fontsize=16, fontweight='bold', color='#2C3E50', pad=10)
    
    # Save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
               facecolor='#F8F9FA', edgecolor='none')
    plt.close()




# Simple AST Visualizer for the real AST - Creates PNG images with proper spacing
def create_ast_png_real(ast_node, filename="ir-ast.png"):
    """
    Create a beautiful PNG image of your real AST with proper node spacing.
    
    Args:
        ast_node: The root AST node (must have .kind, .value, .children)
        filename: Output PNG filename 
    
    Returns:
        str: Path to the created PNG file
    """
    if not ast_node:
        print("❌ No AST node provided")
        return None
        
    print(f"🎨 Creating IR AST visualization: {filename}")
    
    # Colors for different node types
    colors = {
        'program': '#E74C3C',
        'function': '#3498DB', 
        'block': '#2ECC71',
        'declaration': '#F39C12',
        'assignment': '#9B59B6',
        'if': '#E67E22',
        'while': '#1ABC9C',
        'for': '#8E44AD',
        'binary_op': '#27AE60',
        'unary_op': '#D35400',
        'variable': '#FFB6C1',
        'literal': '#FFA07A',
        'default': '#BDC3C7'
    }
    
    # Calculate node width based on label size
    def get_node_width(node):
        label = node.kind
        if hasattr(node, 'value') and node.value:
            label += f"\n{node.value}"
        return max(2.0, 0.5 * len(label))  # Minimum width of 2.0
    
    # First pass: Calculate subtree widths
    subtree_widths = {}
    
    def calc_subtree_width(node):
        if not hasattr(node, 'children') or not node.children:
            width = get_node_width(node)
        else:
            # Calculate total width needed for children with padding
            children_width = 0
            for child in node.children:
                child_width = calc_subtree_width(child)
                children_width += child_width + 1.0  # Add 1.0 as padding between siblings
            
            # Remove extra padding from last child
            if node.children:
                children_width -= 1.0
                
            # Node width is max of its own label width and children's total width
            node_width = get_node_width(node)
            width = max(node_width, children_width)
        
        subtree_widths[id(node)] = width
        return width
    
    # Calculate overall tree width
    total_width = calc_subtree_width(ast_node)
    
    # Second pass: Calculate actual positions
    positions = {}
    
    def calculate_positions(node, level, x_center):
        y_pos = -level * 3.5  # Increased vertical spacing
        positions[id(node)] = (x_center, y_pos)
        
        if hasattr(node, 'children') and node.children:
            # Calculate start position for first child
            child_count = len(node.children)
            
            if child_count == 1:
                # Single child is centered under parent
                calculate_positions(node.children[0], level + 1, x_center)
            else:
                # Calculate positions for multiple children
                total_child_width = sum(subtree_widths[id(child)] for child in node.children)
                total_padding = (child_count - 1) * 2.0  # Increased padding between siblings
                
                # Start position
                current_x = x_center - (total_child_width + total_padding) / 2
                
                for child in node.children:
                    child_width = subtree_widths[id(child)]
                    child_center = current_x + child_width / 2
                    calculate_positions(child, level + 1, child_center)
                    current_x += child_width + 2.0  # Move to next child position with padding
    
    # Start positioning from the center
    calculate_positions(ast_node, 0, total_width / 2)
    
    # Create figure with dynamic sizing based on tree width
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    
    fig_width = max(20, total_width * 1.2)  # Minimum width of 20, but scale up for wide trees
    fig_height = 14
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
    
    # Determine good aspect ratio
    x_coords = [pos[0] for pos in positions.values()]
    y_coords = [pos[1] for pos in positions.values()]
    
    if x_coords and y_coords:
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords) 
        
        if y_range > 0 and x_range > 0:
            # Set aspect ratio that ensures enough vertical space
            aspect_ratio = 0.5 * x_range / y_range
            ax.set_aspect(aspect_ratio)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Set background
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')
    
    # Draw connections and nodes
    def draw_tree(node):
        pos = positions[id(node)]
        
        # Draw connections to children
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                child_pos = positions[id(child)]
                ax.plot([pos[0], child_pos[0]], [pos[1], child_pos[1]], 
                       'k-', linewidth=1.5, alpha=0.6, zorder=1)
        
        # Draw node
        label = node.kind
        if hasattr(node, 'value') and node.value:
            label += f"\n{node.value}"
        
        color = colors.get(node.kind, colors['default'])
        
        # Show temporary name if available
        if hasattr(node, 'temp') and node.temp:
            label += f"\n(temp: {node.temp})"
        
        # Scale box size based on label length
        box_width = min(max(1.2, len(label) * 0.2), 4.0)
        box_height = 0.8 if '\n' in label else 0.5
        
        box = FancyBboxPatch(
            (pos[0] - box_width/2, pos[1] - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.3",
            facecolor=color,
            edgecolor='#2C3E50',
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(box)
        
        ax.text(pos[0], pos[1], label, 
               fontsize=9, fontweight='bold',
               ha='center', va='center',
               color='#2C3E50', zorder=3)
        
        # Draw children
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                draw_tree(child)
    
    draw_tree(ast_node)
    
    # Set limits with margins
    if positions:
        x_coords = [pos[0] for pos in positions.values()]
        y_coords = [pos[1] for pos in positions.values()]
        margin_x = (max(x_coords) - min(x_coords)) * 0.1 + 2.0
        margin_y = (max(y_coords) - min(y_coords)) * 0.1 + 1.0
        ax.set_xlim(min(x_coords) - margin_x, max(x_coords) + margin_x)
        ax.set_ylim(min(y_coords) - margin_y, max(y_coords) + margin_y)
    
    # Add title
    plt.title("Intermediate Representation (IR) AST", 
             fontsize=16, fontweight='bold', color='#2C3E50', pad=10)
    
    # Save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
               facecolor='#F8F9FA', edgecolor='none')
    plt.close()
    
    return filename








black_text = "\033[30m"
red_text = "\033[31m"
green_text = "\033[32m"
yellow_text = "\033[33m"
blue_text = "\033[34m"
magenta_text = "\033[35m"
cyan_text = "\033[36m"
white_text = "\033[37m"

normal_text_start = "\033[0m"
# print(f"{red_text}Red {blue_text}text{normal_text_start}")


class Token:
    def __init__(self, name, lexeme):
        self.name = name
        self.lexeme = lexeme

    def __str__(self):
        return f"{self.name}({self.lexeme})"

    def __repr__(self):
        return str(self)
    
    
    
    
class ASTNode:
    def __init__(self, type, value=None, children=None):
        self.type = type
        self.value = value
        self.children = children if children is not None else []

    def __str__(self, level=0):
        indent = "  " * level
        if self.value is not None:
            rep = f"{indent}{self.type}: {self.value}\n"
        else:
            rep = f"{indent}{self.type}\n"
        for child in self.children:
            rep += child.__str__(level + 1)
        return rep

    def __repr__(self):
        return self.__str__()


def visualize_ast(ast_node, level=0):
    """
    Helper function to visualize the AST in text format.
    """
    indent = "  " * level
    if ast_node.value is not None:
        print(f"{cyan_text}{indent}{ast_node.type}: {ast_node.value}")
        print(f"{white_text}", end="")
        
    else:
        print(f"{cyan_text}{indent}{ast_node.type}")
        print(f"{white_text}", end="")
    for child in ast_node.children:
        visualize_ast(child, level + 1)
    





class LR1Parser:
    def __init__(self):
        self.terminals = []             # this list will hold all the terminals, the tokens that get inputted are all terminals btw. 
        self.non_terminals = []         # this list will hold all the non-terminals, non-terminals are the constructs and the syntactic categories in our language (like we have constructs lke statements, the entire program, the main    function. and we also have several categories for a single construct, like a statement could be a declaration, an if statement or a loop etc.)
        self.productions = {}           # this dictionary will hold all the productions, the keys are the non-terminals and the values are lists of productions for that non-terminal. The productions are the rules that define how the non-terminals can be replaced by terminals and other non-terminals. or simply, what pattern of terminals and non-terminals would be valid in our language
        self.start_symbol = None        # this is the starting symbol of our grammar, the starting symbol is the non-terminal that we start parsing from. It is usually the top-level construct in our language. so it will be program, since that is the top level construct in our language, no matter what code we write.
        self.augmented_start = "S'"     # augmented start symbol is used in LR Parsers to facilitate parsing. now, our normal start symbol (program) only has one production, so we can just use that, but augmenting is useful for other reasons too actually, it defines the acceptance state of the parser which simplifies the algorithm (although you could just do that using program as your start symbol too, but its simply not a best practice)
        self.augmented_grammar = {}     # the same grammar but now we have the new augmented start symbol as well.
        
        self.first_sets = {}           # this dictionary stores the FirstSets of all the symbols in our grammar, this will help during the closure operation phase. 
        self.item_sets = []            # this list will store all the LR(1) item sets, which are the states of the parser. Each item set is a collection of LR(1) items, which are productions with a dot indicating the position of the parser in the production. The LR(1) items also have a lookahead symbol, which is used to determine which production to use when parsing. we will do all this in the closure operation phase btw.
        self.transitions = {}          # this dict will map each state (item set) to the next state based on the input symbol. it will be like {(state, symbol): next_state}
        
        # the following is a bit new, the issue we were facing was that if we encounter any ambiguity like a + b * c (what should be done first?) it would lead us to a shift/reduce conflict. to resolve this these conflicts, we introduce precedence rules into the grammar
        self.precedence = { 
            'power': 5,
            'multiply': 4, 'divide': 4, 'remainder': 4,
            'add': 3, 'subtract': 3,
            'is equal to': 2, 'is not equal to': 2, 'is greater than': 2, 
            'is less than': 2, 'is greater than or equal to': 2, 'is less than or equal to': 2,
            'and': 1, 'or': 1, 'not': 1,
        }

        # ok, so when the tokens are inputted into the parser, they have a name and a lexeme, but the issue is that a single token name could match to multiple possible lexemes, for instance: KEYWORD could be if, else, while, for, etc. So, the parser needs to know which lexeme corresponds to which token name. So, we need a mapping of token names to lexemes. This is done using the following dictionary, where the keys are the token names and the values are lists of lexemes that correspond to that token name.
        self.token_lexemes = {
            'KEYWORD': ['if', 'else', 'while', 'for', 'main', 'function', 'return'],
            'TYPE': ['integer', 'float', 'string', 'boolean'],
            'ARITHMETIC_OP': ['add', 'subtract', 'multiply', 'divide', 'remainder', 'power'],
            'RELATIONAL_OP': ['is equal to', 'is not equal to', 'is greater than', 
            'is less than', 'is greater than or equal to', 'is less than or equal to'],
            'LOGICAL_OP': ['and', 'or', 'not'],
            'ASSIGN_OP': ['equals to'],
        }
        
        # this is the reverse of the above dictionary, where the keys are the lexemes and the values are the token names. This is used to map the lexemes to their corresponding token names when the tokens are inputted into the parser. This will help find token type/name of any lexeme.
        self.lexeme_to_token = {}
        for token_type, lexemes in self.token_lexemes.items():
            for lexeme in lexemes:
                self.lexeme_to_token[lexeme] = token_type
    def define_grammar(self):
        """
        Phase 0: Define grammar for Simple English language with lexeme-specific terminals. We are ganna use the lexemes of the tokens as terminals, except for identifiers and literals, those will be token names, as in the token names will be the terminals there.
        """

        # the list below are the terminals. just to be clear, terminals are the actual lexemes plus token types for variable ones (identifiers, numbers, etc.)
        self.terminals = [
            # Keywords
            'if', 'else', 'while', 'for', 'main', 'function', 'return',
            
            # Types
            'integer', 'float', 'string', 'boolean',
            
            # Operators
            'add', 'subtract', 'multiply', 'divide', 'remainder', 'power',
            'is equal to', 'is not equal to', 'is greater than', 'is less than', 
            'is greater than or equal to', 'is less than or equal to',
            'and', 'or', 'not',
            'equals to',
            
            # Token types for things that vary (can't enumerate)
            'IDENTIFIER', 'NUMBER', 'FLOAT', 'STRING', 'BOOL',
            
            # Punctuation
            'COMMA', 'DOT', 'COLON', 'semicolon',
            '(', ')', '{', '}', '[', ']',
            
            # End marker
            '$'
        ]

        # Non-terminals will simply be hierarchical in nature, so initially we have the entire program as a single NT, then the main function, then the statements in that main function, and then each statement could be a declaration, an assignment, an if case, a while loop, an expression, or a for loop. and furthermore, an expression could be logical, relational, or arithmetic, and inside those expressions, we could have one of the terms as a literal and one as a variable or another term, for that we use Term (like in the slides) and finally, we have the Factor which is the most basic unit of our language, which could be a variable, a number, a float, a string, or a boolean
        self.non_terminals = [
            'Program', 'MainFunction', 'StatementList', 'Statement', 
            'Declaration', 'Assignment', 'IfStatement', 'WhileStatement', 'ForStatement', 
            'Expression', 'LogicalExpr', 'RelationalExpr', 'ArithmeticExpr', 
            'Term', "PowerExpr", 'Factor', 'Condition', 'Type', "ReturnStatement"
            # 'ElseIfList', 'ElseIf'
        ]

        # Productions
        self.productions = {
            'Program': [['MainFunction']],  # a program would have exactly one main function
            
            'MainFunction': [
                ['Type', 'main', '(', ')', '{', 'StatementList', '}']   # the main function will have exactly this format
            ],
            
            'Type': [
                ['integer'], ['float'], ['string'], ['boolean']
            ],
            
            'StatementList': [
                ['Statement'],  # a statement inside our main function could be a single statement
                ['StatementList', 'Statement']  # or it could be a list of statements, left recursion might happen here but we don't care since this is an LR parser
            ],

            'Statement': [              # there could be several types of statements
                ['Declaration', 'semicolon'],   # a simple declaration statement, see declaration NT rules for details
                ['Assignment', 'semicolon'],    # an assignment statement (you can declare and assign in the same statement, see declaration rules)
                ['IfStatement'],        # an if statement, see if statement NT rules for details
                ['WhileStatement'],     # a while loop, see while statement NT rules for details
                ['ForStatement'],            # a for loop, see for statement NT rules for details
                ['ReturnStatement']

                
            ],

            'Declaration': [
                ['Type', 'IDENTIFIER', 'equals to', 'Expression'],  # this is a declaration statement, it declares a variable of a certain type and assigns it an initial value
                ['Type', 'IDENTIFIER'] # this is a declaration statement without an initial value, it just declares a variable of a certain type
            ],

            'Assignment': [
                ['IDENTIFIER', 'equals to', 'Expression']   # this is an assignment statement, it assigns a value to a variable
            ],

            'IfStatement': [    # we don't bother with else ifs.
                ['if', '(', 'Condition', ')', '{', 'StatementList', '}'],
                ['if', '(', 'Condition', ')', '{', 'StatementList', '}', 'else', '{', 'StatementList', '}']
            ],
            
            # 'IfStatement': [
            # ['if', '(', 'Condition', ')', '{', 'StatementList', '}'],                                    # Simple if
            # ['if', '(', 'Condition', ')', '{', 'StatementList', '}', 'else', '{', 'StatementList', '}'], # if-else
            # ['if', '(', 'Condition', ')', '{', 'StatementList', '}', 'ElseIfList'],                     # if with else-ifs
            # ['if', '(', 'Condition', ')', '{', 'StatementList', '}', 'ElseIfList', 'else', '{', 'StatementList', '}'] # if with else-ifs and final else
            # ],

            # 'ElseIfList': [
            # ['ElseIf'],
            # ['ElseIfList', 'ElseIf']
            # ],

            # 'ElseIf': [
            # ['else', 'if', '(', 'Condition', ')', '{', 'StatementList', '}']
            # ],

            'WhileStatement': [ # while loop syntax is just like C, nothing complicated
                ['while', '(', 'Condition', ')', '{', 'StatementList', '}']
            ],

            'ForStatement': [   # we use the declaration NT here, if the user simply declares a variable without assigning it a value, then that would cause a semantic error, we will deal with this in the semantic analysis phase
                ['for', '(', 'Declaration', 'semicolon', 'Condition', 'semicolon', 'Assignment', ')', '{', 'StatementList', '}']
            ],

            'ReturnStatement': [
                ['return', 'Expression', 'semicolon'],
                ['return', 'semicolon']  # For void returns
            ],


            'Expression': [ # we will nest arithmetic exp in relational exp and relational exp in logical exp, this is done to deal with the precedence of operators. the more nested the exp is, the higher the precedence.
            # for example: a + b == c AND d < e is one big expression that contains arithmetic expressions and relational expressions and logical expressions
            # we first solve the arithmetic part, then the relational part, and finally the logical part.
            # so the logical exp will be the top-level construct in an expression. that is why, we only have onr production for Expression, which is a LogicalExpr:
                ['LogicalExpr']
            ],
            
            'LogicalExpr': [
                ['RelationalExpr'],
                ['LogicalExpr', 'and', 'RelationalExpr'],   # logical AND
                ['LogicalExpr', 'or', 'RelationalExpr'],    # logical OR
                ['not', 'RelationalExpr']                   # logical NOT
            ],
            
            'RelationalExpr': [
                ['ArithmeticExpr'],
                ['ArithmeticExpr', 'is equal to', 'ArithmeticExpr'],
                ['ArithmeticExpr', 'is not equal to', 'ArithmeticExpr'],
                ['ArithmeticExpr', 'is greater than', 'ArithmeticExpr'],
                ['ArithmeticExpr', 'is less than', 'ArithmeticExpr'],
                ['ArithmeticExpr', 'is greater than or equal to', 'ArithmeticExpr'],
                ['ArithmeticExpr', 'is less than or equal to', 'ArithmeticExpr']
            ],
            
            'ArithmeticExpr': [
                ['Term'],
                ['ArithmeticExpr', 'add', 'Term'],
                ['ArithmeticExpr', 'subtract', 'Term']
            ],
            
            'Term': [
                ['PowerExpr'],                      
                ['Term', 'multiply', 'PowerExpr'],  
                ['Term', 'divide', 'PowerExpr'],
                ['Term', 'remainder', 'PowerExpr']
            ],

            'PowerExpr': [
                ['Factor'],
                ['Factor', 'power', 'PowerExpr']    # Right associative: 2^3^4 = 2^(3^4)
            ],
            
            'Factor': [
                ['IDENTIFIER'],
                ['NUMBER'],
                ['FLOAT'],
                ['STRING'],
                ['BOOL'],
                ['(', 'Expression', ')']
            ],

            'Condition': [
                ['Expression']
            ]
        }

        self.start_symbol = 'Program'

        # Debug output
        # print("----- PHASE 0: GRAMMAR DEFINITION -----")
        # print(f"==========================================")
        # print(f"Terminals count: {len(self.terminals)}")
        # print(f"Terminals: {self.terminals}")
        # print(f"==========================================")
        # print(f"Non-terminals count: {len(self.non_terminals)}")
        # print(f"Non-terminals: {self.non_terminals}")
        # print(f"==========================================")
        # print(f"Start Symbol: {self.start_symbol}")
        # print("ALL Production Rules:")
        # print(f"Productions count: {len(self.productions)}")
        # for nt in self.non_terminals:
        #     if nt in self.productions:
        #         for i, prod in enumerate(self.productions[nt]):
        #             print(f"{nt} -> {' '.join(prod)}")
        # print("\n")
        # print(f"==========================================\n")

        return (self.terminals, self.non_terminals, self.productions, self.start_symbol)

        
    def augment_grammar(self):
        """
        Phase 1-A: Augment grammar with S' -> Program. Just a single production rule.
        """
        self.augmented_grammar = {}
        self.augmented_grammar[self.augmented_start] = [[self.start_symbol]]    # S' -> Program
        for k, v in self.productions.items():
            self.augmented_grammar[k] = copy.deepcopy(v) # deepcopy is used to avoid shallow copy issues, since we are modifying the grammar in place.
        if self.augmented_start not in self.non_terminals: self.non_terminals.append(self.augmented_start)

        # print("----- PHASE 1-A: AUGMENTED GRAMMAR -----")
        # print(f"Augmented Start Symbol: {self.augmented_start}")
        # print("Augmented Production Rules:")
        # print(f"{self.augmented_start} -> {' '.join(self.augmented_grammar[self.augmented_start][0])}")
        # for nt in self.productions:
        #     for i, prod in enumerate(self.productions[nt]):
        #         print(f"{nt} -> {' '.join(prod)}")
            
        # print("\n")
        # print("==========================================\n")

        return self.augmented_grammar
    def make_lr1_item(self, non_terminal, production, dot_position, lookahead):
        return (non_terminal, tuple(production), dot_position, lookahead)   # its like this: [NT -> production(as a tuple), intIndex of dot, lookahead]. we will pass this whole thing as as arg and display it using the method below

    def display_lr1_item(self, item):
        lhs, rhs, dot_pos, lookahead = item # extract the values from the tuple
        rhs_with_dot = list(rhs)    # the NT part
        rhs_with_dot.insert(dot_pos, '•')   # place the dot in the rhs of the production
        return f"[{lhs} -> {' '.join(rhs_with_dot)}, {lookahead}]"  # return it all as a nice f-string

    def get_initial_item_set(self): # now lets make our first LRItem using the augmented productions. it should be a simple: [s' -> DOT program, $ ]
        initial_prod = self.augmented_grammar[self.augmented_start][0]  # S' -> Program retrieved from the dict
        initial_item = self.make_lr1_item(self.augmented_start, initial_prod, 0, '$') # add the dot and lookahead to make the LRItem
        # print("----- PHASE 1-B: INITIAL LR(1) ITEM -----")
        # print(f"Initial Item: {self.display_lr1_item(initial_item)}")
        # print("\n")
        # print("==========================================")
        return [initial_item]
        
    def compute_first_sets(self):
        """
        Phase 2: Compute FIRST sets for terminals and non-terminals
        """
        self.first_sets = {}

        for t in self.terminals:
            self.first_sets[t] = {t}
        for nt in self.non_terminals:
            self.first_sets[nt] = set()

        changed = True
        while changed:
            changed = False
            for nt in self.non_terminals:
                for prod in self.augmented_grammar.get(nt, []):
                    if not prod:
                        if 'ε' not in self.first_sets[nt]:
                            self.first_sets[nt].add('ε')
                            changed = True
                        continue
                    for symbol in prod:
                        before = len(self.first_sets[nt])
                        self.first_sets[nt].update(s for s in self.first_sets.get(symbol, set()) if s != 'ε')
                        after = len(self.first_sets[nt])
                        if after > before:
                            changed = True
                        if 'ε' not in self.first_sets.get(symbol, set()):
                            break
                    else:
                        if 'ε' not in self.first_sets[nt]:
                            self.first_sets[nt].add('ε')
                            changed = True
        
        # print("----- PHASE 2: FIRST SETS -----")
        # for nt in ['Program', 'MainFunction', 'Statement', 'Expression']:
        #     print(f"FIRST({nt}) = {self.first_sets[nt]}")
        # print("\n")
        return self.first_sets
    def compute_first_of_string(self, symbols):
        """
        Compute FIRST set of a string of grammar symbols.
        """
        if not symbols:
            return {'ε'}
        result = set()
        for symbol in symbols:
            result.update(s for s in self.first_sets.get(symbol, set()) if s != 'ε')
            if 'ε' not in self.first_sets.get(symbol, set()):
                break
        else:
            result.add('ε')
        return result
    def closure(self, items):
        """
        Phase 3: Compute closure of LR(1) item set.
        """
        closure_set = list(items)   # this is the start of the fission process, we start off with one item in our state
        added = True                # we keep going until no new items can be added, that is when a state is completed
        while added:
            added = False
            for item in closure_set[:]:
                lhs, rhs, dot_pos, lookahead = item
                if dot_pos >= len(rhs): # parsed through the entire handle? if so then move to next item
                    continue
                next_sym = rhs[dot_pos] # get the next symbol
                if next_sym in self.non_terminals:  # if it is a non-terminal
                    beta = rhs[dot_pos + 1:] if dot_pos + 1 < len(rhs) else [] # get the rest of the handle
                    lookaheads = self.compute_first_of_string(list(beta) + [lookahead]) # combine the rest of the handle with the lookahead and take first of that, this will be the new lookahead
                    for prod in self.augmented_grammar.get(next_sym, []):   # get the productions of the next_sym
                        for la in lookaheads:
                            if la == 'ε':
                                la = lookahead
                            new_item = self.make_lr1_item(next_sym, prod, 0, la)    # combine the production and the lookahead and we have our new item
                            if new_item not in closure_set: # if its not a duplicate, add to our state/set
                                closure_set.append(new_item)
                                added = True
        return closure_set
    def goto(self, items, symbol):
        """
        Phase 4: Compute goto on symbol from items.
        """
        next_items = []
        for item in items:
            lhs, rhs, dot_pos, lookahead = item
            if dot_pos < len(rhs) and rhs[dot_pos] == symbol:
                next_items.append(self.make_lr1_item(lhs, rhs, dot_pos + 1, lookahead)) # moving the marker one step ahead
        return self.closure(next_items) if next_items else []   # new state after moving the marker
    def is_same_item_set(self, set1, set2):
        """
        Check if two LR(1) item sets are identical.
        """
        if len(set1) != len(set2):  # a simple method to check if two states (sets of items) are the same (if they are, we should merge)
            return False
        set1_items = {(lhs, rhs, dot, la) for lhs, rhs, dot, la in set1}
        set2_items = {(lhs, rhs, dot, la) for lhs, rhs, dot, la in set2}
        return set1_items == set2_items
    def build_canonical_collection(self):
        """
        Phase 5: Build canonical collection of LR(1) item sets and transitions.
        """
        initial_items = self.get_initial_item_set() # our S' -> Program
        initial_closure = self.closure(initial_items)   # the first fission, the first state is made here, it would be something like:
        self.item_sets = [initial_closure]  # the first state, item_sets will ultimately contain all states
        self.transitions = {}

        all_symbols = self.terminals + self.non_terminals

        idx = 0
        while idx < len(self.item_sets):
            state = self.item_sets[idx] # current state (initially it would be the first state)
            for sym in all_symbols:
                goto_set = self.goto(state, sym)
                if goto_set:
                    found = False
                    for i, existing_state in enumerate(self.item_sets):
                        if self.is_same_item_set(goto_set, existing_state):
                            self.transitions[(idx, sym)] = i
                            found = True
                            break
                    if not found:
                        self.item_sets.append(goto_set)
                        self.transitions[(idx, sym)] = len(self.item_sets) - 1
            idx += 1

        # print("----- PHASE 5: CANONICAL COLLECTION OF LR(1) ITEMS -----")
        # print(f"Total states: {len(self.item_sets)}")
        # print(f"Total transitions: {len(self.transitions)}")
        # print("\n")

        return (self.item_sets, self.transitions)
        
    def get_precedence(self, token_value):
        """
        Get precedence level for operator.
        """
        if token_value in self.precedence:
            return self.precedence[token_value]
        return 0  # Default precedence for non-operators
        
    def resolve_conflict(self, state_idx, symbol, action1, action2):
        """
        Resolve conflict based on operator precedence.
        """
        state = self.item_sets[state_idx]
        
        # Check if this is a shift-reduce conflict
        if action1[0] == 'shift' and action2[0] == 'reduce':
            shift_action, reduce_action = action1, action2
        elif action1[0] == 'reduce' and action2[0] == 'shift':
            reduce_action, shift_action = action1, action2
        else:
            # For reduce-reduce conflicts, prefer the production with the non-terminal that comes first
            # in our grammar (more specific productions first)
            nt1, idx1 = action1[1]
            nt2, idx2 = action2[1]
            
            # Order of precedence for statements
            statement_order = {
                'IfStatement': 1,
                'WhileStatement': 2,
                'ForStatement': 3,
            }
            
            if nt1 in statement_order and nt2 in statement_order:
                if statement_order[nt1] < statement_order[nt2]:
                    return action1
                else:
                    return action2
            
            # Default behavior for other reduce-reduce conflicts
            if self.non_terminals.index(nt1) < self.non_terminals.index(nt2):
                return action1
            else:
                return action2
        
        # For shift-reduce conflicts involving operators, use precedence
        reduce_nt, reduce_idx = reduce_action[1]
        reduce_prod = self.augmented_grammar[reduce_nt][reduce_idx]
        
        # Look for operators in the production and symbol
        reduce_op = None
        for i, sym in enumerate(reduce_prod):
            if sym in self.precedence:
                reduce_op = sym
                break
        
        shift_op = symbol if symbol in self.precedence else None
        
        if reduce_op and shift_op:
            # Compare precedence
            reduce_prec = self.get_precedence(reduce_op)
            shift_prec = self.get_precedence(shift_op)
            
            if shift_prec > reduce_prec:
                return shift_action
            else:
                return reduce_action
        
        # Default resolution: prefer shift over reduce
        return shift_action
        
    def build_parsing_tables(self):
        """
        Phase 6: Build ACTION and GOTO tables from canonical collection.
        ACTION: (state, terminal) -> ('shift', s) | ('reduce', (nt, prod_idx)) | ('accept', None)
        GOTO: (state, non-terminal) -> state
        Apply conflict resolution.
        """
        self.action_table = {}
        self.goto_table = {}
        conflicts_before_resolution = []
        conflicts_after_resolution = []

        for state_idx, state in enumerate(self.item_sets):
            for item in state:
                lhs, rhs, dot_pos, lookahead = item

                # If dot not at end, possibly shift
                if dot_pos < len(rhs):
                    next_sym = rhs[dot_pos]
                    if next_sym in self.terminals:
                        if (state_idx, next_sym) in self.transitions:
                            next_state = self.transitions[(state_idx, next_sym)]
                            action = ('shift', next_state)
                            if (state_idx, next_sym) in self.action_table:
                                existing = self.action_table[(state_idx, next_sym)]
                                if existing != action:
                                    conflicts_before_resolution.append(((state_idx, next_sym), existing, action))
                                    # Apply conflict resolution
                                    resolved = self.resolve_conflict(state_idx, next_sym, existing, action)
                                    if resolved != self.action_table[(state_idx, next_sym)]:
                                        conflicts_after_resolution.append(((state_idx, next_sym), existing, action, resolved))
                                    self.action_table[(state_idx, next_sym)] = resolved
                            else:
                                self.action_table[(state_idx, next_sym)] = action

                else:
                    # Dot at end: reduce or accept
                    if lhs == self.augmented_start and rhs == (self.start_symbol,) and lookahead == '$':
                        self.action_table[(state_idx, '$')] = ('accept', None)
                    else:
                        # Find production index
                        prod_num = None
                        for nt, prods in self.augmented_grammar.items():
                            for i, prod in enumerate(prods):
                                if nt == lhs and tuple(prod) == rhs:
                                    prod_num = (nt, i)
                                    break
                            if prod_num is not None:
                                break
                        action = ('reduce', prod_num)
                        if (state_idx, lookahead) in self.action_table:
                            existing = self.action_table[(state_idx, lookahead)]
                            if existing != action:
                                conflicts_before_resolution.append(((state_idx, lookahead), existing, action))
                                # Apply conflict resolution
                                resolved = self.resolve_conflict(state_idx, lookahead, existing, action)
                                if resolved != self.action_table[(state_idx, lookahead)]:
                                    conflicts_after_resolution.append(((state_idx, lookahead), existing, action, resolved))
                                self.action_table[(state_idx, lookahead)] = resolved
                        else:
                            self.action_table[(state_idx, lookahead)] = action
            
            # Build GOTO entries for non-terminals
            for nt in self.non_terminals:
                if (state_idx, nt) in self.transitions:
                    self.goto_table[(state_idx, nt)] = self.transitions[(state_idx, nt)]

        # print("----- PHASE 6: PARSING TABLES -----")
        # print(f"Total ACTION entries: {len(self.action_table)}")
        # print(f"Total GOTO entries: {len(self.goto_table)}")
        # print(f"Conflicts detected before resolution: {len(conflicts_before_resolution)}")
        # print(f"Conflicts remaining after resolution: {len(conflicts_after_resolution)}")
        # print("\n")
        
        return self.action_table, self.goto_table, conflicts_after_resolution
    def match_token_to_terminal(self, token):
        """
        Match a token to the appropriate terminal in our grammar.
        For tokens with specific lexemes like KEYWORD/TYPE/etc, use the lexeme.
        For others like IDENTIFIER/NUMBER, use the token name.
        """
        token_name = token.name
        token_lexeme = token.lexeme
        
        # If this is a token type with lexemes we care about
        if token_name in self.token_lexemes:
            # Use the lexeme as the terminal if it's in our grammar
            if token_lexeme in self.terminals:
                return token_lexeme
        
        # For variable content types, use the token name
        if token_name in self.terminals:
            return token_name
            
        # Special case for brackets, parens and other punctuation
        if token_name in ['LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET', 'SEMI']:
            # Map token names to the actual symbols in our grammar
            token_sym_map = {
                'LPAREN': '(', 'RPAREN': ')', 
                'LBRACE': '{', 'RBRACE': '}', 
                'LBRACKET': '[', 'RBRACKET': ']',
                'SEMI': 'semicolon'
            }
            return token_sym_map.get(token_name, token_name)
            
            # If we can't match it properly, return the token name but log a warning
        print(f"Warning: Couldn't match token {token} to a terminal in the grammar")
        return token_name
    def parse_input(self, tokens, log_filename='parsing_log.txt', perform_semantic_analysis=True):
        """
        Parse a list of tokens and optionally perform simplified semantic analysis.
        """
        if not tokens or tokens[-1].name != '$':
            tokens.append(Token('$', '$'))  # Ensure end marker

        stack = [0]  # state stack
        semantic_stack = []  # semantic nodes (AST nodes)
        token_index = 0
        log_lines = []
        accepted = False
        error_msg = None
        ast_nodes = []  # List to collect all AST nodes

        log_lines.append("Parsing started")
        log_lines.append(f"Tokens: {[f'{t.name}({t.lexeme})' for t in tokens]}\n")

        while True:
            state = stack[-1]
            current_token = tokens[token_index]

            # Match token to grammar terminal
            terminal = self.match_token_to_terminal(current_token)

            action_key = (state, terminal)
            action = self.action_table.get(action_key, None)

            log_lines.append(f"Stack: {stack}")
            log_lines.append(f"Semantic stack depth: {len(semantic_stack)}")
            log_lines.append(f"Next token: {current_token.name}({current_token.lexeme}) -> terminal: {terminal}")

            if action is None:
                error_msg = f"No action for state {state} and terminal {terminal}"
                log_lines.append(f"ERROR: {error_msg}")
                expected_terminals = [term for (s, term) in self.action_table.keys() if s == state]
                log_lines.append(f"Expected one of: {expected_terminals}")
                log_lines.append(f"Error at token index {token_index}: {current_token.name}({current_token.lexeme})")

                # Add more context for error reporting
                if token_index > 0:
                    prev_token = tokens[token_index - 1]
                    log_lines.append(f"Previous token: {prev_token.name}({prev_token.lexeme})")

                # Check for unbalanced braces/parentheses
                open_count = sum(1 for t in tokens[:token_index] if t.lexeme in ["(", "{", "["])
                close_count = sum(1 for t in tokens[:token_index] if t.lexeme in [")", "}", "]"])
                if open_count != close_count:
                    brace_error = f"Possible syntax error: Unbalanced brackets/braces/parentheses (open: {open_count}, closed: {close_count})"
                    log_lines.append(brace_error)
                    if error_msg:
                        error_msg += f". {brace_error}"

                break

            act_type, act_val = action
            log_lines.append(f"Action: {act_type} {act_val if act_val is not None else ''}")

            if act_type == 'shift':
                # Store both token name and lexeme in the AST node
                node = ASTNode(current_token.name, current_token.lexeme)
                semantic_stack.append(node)
                ast_nodes.append(node)  # Add leaf node to our collection
                stack.append(act_val)
                token_index += 1

            elif act_type == 'reduce':
                nt, prod_idx = act_val
                prod = self.augmented_grammar[nt][prod_idx]
                pop_len = len(prod)

                # Pop states and semantic nodes accordingly
                for _ in range(pop_len):
                    stack.pop()
                children = []
                for _ in range(pop_len):
                    if semantic_stack:
                        children.insert(0, semantic_stack.pop())

                # Special handling for certain non-terminals to improve AST structure
                if nt == 'ArithmeticExpr' and len(children) == 3:  # Binary operation
                    operator_node = children[1]
                    new_node = ASTNode('BinaryOperation', operator_node.value, [children[0], children[2]])
                elif nt == 'RelationalExpr' and len(children) == 3:  # Relational operation
                    operator_node = children[1]
                    new_node = ASTNode('RelationalOperation', operator_node.value, [children[0], children[2]])
                elif nt == 'Assignment' and len(children) == 3:  # Assignment
                    new_node = ASTNode('Assignment', None, [children[0], children[2]])
                elif nt == 'Declaration' and len(children) >= 2:  # Variable declaration
                    new_node = ASTNode('Declaration', None, children)
                elif nt == 'LogicalExpr' and len(children) == 3:  # Logical operation
                    operator_node = children[1]
                    new_node = ASTNode('LogicalOperation', operator_node.value, [children[0], children[2]])
                elif nt == 'ForLoop' and len(children) >= 7:  # For loop structure
                    new_node = ASTNode('ForLoop', None, children)
                elif nt == 'IfStatement' and len(children) >= 5:  # If statement
                    new_node = ASTNode('IfStatement', None, children)
                elif nt == 'Program':  # Top-level program
                    new_node = ASTNode('Program', None, children)
                elif nt == 'Block':  # Code block
                    new_node = ASTNode('Block', None, children)
                else:
                    new_node = ASTNode(nt, None, children)

                semantic_stack.append(new_node)
                ast_nodes.append(new_node)  # Add non-terminal node to our collection

                top_state = stack[-1]
                goto_state = self.goto_table.get((top_state, nt), None)
                if goto_state is None:
                    error_msg = f"No GOTO entry for state {top_state} and non-terminal {nt}"
                    log_lines.append(f"ERROR: {error_msg}")
                    break
                stack.append(goto_state)

            elif act_type == 'accept':
                log_lines.append("Input accepted!")
                accepted = True
                break

            log_lines.append('')

        # Write parsing log
        with open(log_filename, 'w') as f:
            for line in log_lines:
                f.write(line + '\n')
        print(f"Parsing log saved to '{log_filename}'")

        parsing_success = accepted and semantic_stack
        semantic_success = None
        semantic_errors = []
        
        # Perform simple semantic analysis if requested and parsing was successful
        if perform_semantic_analysis and parsing_success:
            print(f"\n{blue_text}=== SEMANTIC ANALYSIS ===")
            root_ast = semantic_stack[0] if semantic_stack else None
            
            if root_ast:
                # Perform simplified semantic analysis
                semantic_success, semantic_errors = self.simple_semantic_analysis(root_ast)
                
                # Print results
                if semantic_errors:
                    print(f"{red_text}Found {len(semantic_errors)} semantic errors:")
                    for error in semantic_errors:
                        print(f"- {error}")
                    print(f"{white_text}", end='')  # Reset color
                else:
                    print(f"{green_text}✓ No semantic errors found!{white_text}")
            else:
                print("No AST root found for semantic analysis")
                semantic_success = False
        
        if parsing_success:
            # Return the complete AST and success status
            return semantic_stack, True, semantic_success, None, semantic_errors
        else:
            # Return whatever AST nodes we've collected so far for debugging
            error_msg = error_msg or "Parsing failed: Unknown error"
            return ast_nodes, False, semantic_success, error_msg, semantic_errors
    
    def simple_semantic_analysis(self, ast):
        """
        Simplified semantic analysis that checks for:
        1. Type mismatches in declarations and assignments
        2. Uninitialized variables
        3. Undeclared variables
        
        Returns: (success, error_list)
        """
        errors = []
        symbol_table = {}  # {name: {'type': type, 'initialized': bool}}
        
        def analyze_node(node):
            if node.type == 'Declaration':
                # Find type and identifier
                var_type = None
                var_name = None
                has_init = False
                
                for child in node.children:
                    if child.type == 'Type':
                        if child.children and child.children[0].type == 'TYPE':
                            var_type = child.children[0].value
                    elif child.type == 'IDENTIFIER':
                        var_name = child.value
                    elif child.type == 'ASSIGN_OP':
                        has_init = True
                
                if var_type and var_name:
                    # Check for redeclaration
                    if var_name in symbol_table:
                        errors.append(f"Variable '{var_name}' already declared")
                    else:
                        # Add to symbol table
                        symbol_table[var_name] = {'type': var_type, 'initialized': has_init}
                        
                        # Check for type mismatch in initialization if present
                        if has_init and len(node.children) > 3:  # Type, ID, equals, Expr
                            expr_type = get_expr_type(node.children[3])
                            if not types_compatible(var_type, expr_type):
                                errors.append(f"Type mismatch: cannot assign {expr_type} to {var_type} variable '{var_name}'")
            
            elif node.type == 'Assignment':
                if len(node.children) >= 2:
                    var_node = node.children[0]
                    expr_node = node.children[-1]  # Last child should be expression
                    
                    if var_node.type == 'IDENTIFIER':
                        var_name = var_node.value
                        # Check if variable is declared
                        if var_name not in symbol_table:
                            errors.append(f"Variable '{var_name}' used but not declared")
                        else:
                            # Mark as initialized
                            symbol_table[var_name]['initialized'] = True
                            
                            # Check type compatibility
                            var_type = symbol_table[var_name]['type']
                            expr_type = get_expr_type(expr_node)
                            if not types_compatible(var_type, expr_type):
                                errors.append(f"Type mismatch: cannot assign {expr_type} to {var_type} variable '{var_name}'")
            
            elif node.type == 'IDENTIFIER':
                var_name = node.value
                # Check if variable is declared
                if var_name not in symbol_table:
                    errors.append(f"Variable '{var_name}' used but not declared")
                # Check if variable is initialized
                elif not symbol_table[var_name]['initialized']:
                    errors.append(f"Variable '{var_name}' used before initialization")
            
            # Recurse through children
            if hasattr(node, 'children'):
                for child in node.children:
                    analyze_node(child)
        
        def get_expr_type(node):
            """Determine the type of an expression node"""
            if node.type == 'NUMBER':
                return 'integer'
            elif node.type == 'FLOAT':
                return 'float'
            elif node.type == 'STRING':
                return 'string'
            elif node.type == 'BOOL':
                return 'boolean'
            elif node.type == 'IDENTIFIER':
                if node.value in symbol_table:
                    return symbol_table[node.value]['type']
                return 'unknown'
            elif node.type in ['RelationalOperation', 'LogicalOperation']:
                return 'boolean'
            elif node.type == 'BinaryOperation':
                # For simplicity, assume the result type matches the left operand
                if node.children:
                    return get_expr_type(node.children[0])
            
            # For other nodes, look at children
            if hasattr(node, 'children') and node.children:
                for child in node.children:
                    child_type = get_expr_type(child)
                    if child_type and child_type != 'unknown':
                        return child_type
            
            return 'unknown'
        
        def types_compatible(target_type, source_type):
            """Check if source_type can be assigned to target_type"""
            if target_type == source_type:
                return True
            # Allow integer to float promotion
            if target_type == 'float' and source_type == 'integer':
                return True
            return False
        
        # Start analysis from the root
        analyze_node(ast)
        
        success = len(errors) == 0
        return success, errors
        
    
    def test_program(self, tokens, test_name, description, log_filename=None, create_visualization=False, visualization_filename=None):
        """
        Test a program with the parser and display results in a clean format with improved error messages.
        
        Args:
            tokens (list): List of Token objects representing the program
            test_name (str): Name of the test (e.g., "TYPE MISMATCH")
            description (str): Short description of what's being tested
            log_filename (str, optional): Filename for parsing log. If None, generates one based on test name
            create_visualization (bool, optional): Whether to create AST visualization image
            visualization_filename (str, optional): Filename for AST visualization. If None, generates one based on test name
            
        Returns:
            tuple: (ast_nodes, parsing_success, semantic_success) for additional processing if needed
        """
        # Generate default filenames if not provided
        if not log_filename:
            log_filename = f"test-{test_name.lower().replace(' ', '-')}-log.txt"
        
        if create_visualization and not visualization_filename:
            visualization_filename = f"ast-{test_name.lower().replace(' ', '-')}.png"
            
        # Print test header
        print(f"\n{magenta_text}=== TEST: {test_name} ===")
        print(f"Testing: {description}")
        print(f"{normal_text_start}")

        # Run the parser with semantic analysis
        ast_nodes, parsing_success, semantic_success, syntax_error_msg, semantic_errors = self.parse_input(
            tokens, log_filename, perform_semantic_analysis=True
        )
        
        # Print parsing results
        if parsing_success:
            print(f"\n{green_text}Parsing: SUCCESS{normal_text_start}")
            
            # Print semantic analysis results
            if semantic_success:
                print(f"{green_text}Semantic Analysis: SUCCESS{normal_text_start}")
            else:
                print(f"{red_text}Semantic Analysis: FAILED{normal_text_start}")
                # print(f"{red_text}Semantic Errors:{normal_text_start}")
                # for error in semantic_errors:
                #     print(f" - {error}")
            
            # Display AST/parseTree
            print(f"\n{cyan_text}Parse Tree Structure:{normal_text_start}")
            root_node = ast_nodes[0] if ast_nodes else None
            if root_node:
                visualize_ast(root_node)
                if parsing_success and semantic_success:
                    print(f"\n{blue_text}=== INTERMEDIATE REPRESENTATION ==={normal_text_start}")
                    true_ast, tac_code = self.generate_ir(root_node)
                # Display real AST
                    print(f"\n{blue_text}Abstract Syntax Tree:{normal_text_start}")
                    self.visualize_ast_real(true_ast)
                # Display Three-Address Code
                    self.display_tac(tac_code)

                # Create AST visualization for IR if requested
                if create_visualization:
                # Parse tree visualization (original)
                    create_ast_png(root_node, visualization_filename)
                    print(f"\n{blue_text}Parse Tree visualization saved to: {visualization_filename}{normal_text_start}")
                    
                    if parsing_success and semantic_success:
                    # IR AST visualization (new)
                        ir_viz_filename = f"ir-{test_name.lower().replace(' ', '-')}.png"
                        create_ast_png_real(true_ast, ir_viz_filename)
                        print(f"\n{blue_text}IR AST visualization saved to: {ir_viz_filename}{normal_text_start}")
            else:
                print("No AST generated.")
        else:
            print(f"\n{red_text}Parsing: FAILED{normal_text_start}")
            
            # Improve error message for common syntax errors
            improved_error = self._provide_helpful_error_message(syntax_error_msg, tokens)
            print(f"{yellow_text}Error details:{normal_text_start}")
            print(f" - {improved_error}")
            
            # Show context around the error
            self._show_error_context(tokens)
        
        # Print separator
        print(f"\n{normal_text_start}" + "="*60)
        if parsing_success and semantic_success:
            true_ast, tac_code = self.generate_ir(root_node)
        else:
            true_ast, tac_code = None, []
        return true_ast, tac_code
    
    def _provide_helpful_error_message(self, error_msg, tokens):
        """
        Convert technical error messages into more helpful, user-friendly messages.
        
        Args:
            error_msg (str): The original error message
            tokens (list): List of tokens being parsed
            
        Returns:
            str: An improved error message
        """
        if not error_msg:
            return "Unknown parsing error"
            
        # Extract token information if "No action for state X and terminal Y" pattern
        if "No action for state" in error_msg and "terminal" in error_msg:
            state_match = re.search(r"state (\d+)", error_msg)
            terminal_match = re.search(r"terminal (\w+|\{|\}|\(|\)|\[|\]|\$)", error_msg)
            
            if state_match and terminal_match:
                state = state_match.group(1)
                terminal = terminal_match.group(1)
                
                # Check for common syntax errors
                if terminal == '}':
                    # Missing semicolon is a common error before closing braces
                    return "Possible syntax error: Missing semicolon before closing brace"
                
                elif terminal == ')':
                    return "Possible syntax error: Unexpected closing parenthesis or missing semicolon"
                
                elif terminal == '$':
                    return "Possible syntax error: Unexpected end of input, check for missing tokens"
                    
                # Check for unbalanced braces separately
                if "Unbalanced brackets/braces/parentheses" in error_msg:
                    # Extract the counts
                    open_match = re.search(r"open: (\d+)", error_msg)
                    close_match = re.search(r"closed: (\d+)", error_msg)
                    
                    if open_match and close_match:
                        open_count = int(open_match.group(1))
                        close_count = int(close_match.group(1))
                        
                        if open_count > close_count:
                            return f"Syntax error: Missing {open_count - close_count} closing brace(s) or parenthesis/parentheses"
                        else:
                            return f"Syntax error: {close_count - open_count} too many closing brace(s) or parenthesis/parentheses"
        
        return error_msg
    
    def _show_error_context(self, tokens):
        """
        Show context around the error to help identify the issue.
        
        Args:
            tokens (list): List of tokens being parsed
        """
        # Find where parsing likely failed (often near the end of processed tokens)
        error_vicinity = len(tokens) - 5  # Assume error is near the end
        start_idx = max(0, error_vicinity - 5)
        end_idx = min(len(tokens), error_vicinity + 5)
        
        print(f"\n{yellow_text}Context around error:{normal_text_start}")
        print("..." if start_idx > 0 else "")
        
        for i in range(start_idx, end_idx):
            if i == error_vicinity:
                print(f"{red_text}→ {i}: {tokens[i]}{normal_text_start}")
            else:
                print(f"  {i}: {tokens[i]}")
        
        print("..." if end_idx < len(tokens) else "")
        
        # Provide additional hint for common token sequences
        self._suggest_fixes(tokens, error_vicinity)
    
    def _suggest_fixes(self, tokens, error_idx):
        """
        Suggest possible fixes based on common error patterns.
        
        Args:
            tokens (list): List of tokens being parsed
            error_idx (int): Index where error likely occurred
        """
        if error_idx >= len(tokens) or error_idx < 1:
            return
        
        # Check for patterns that often indicate missing semicolons
        current_token = tokens[error_idx]
        prev_token = tokens[error_idx - 1] if error_idx > 0 else None
        
        if not prev_token:
            return
            
        # Missing semicolon patterns
        if (prev_token.name in ['IDENTIFIER', 'NUMBER', 'FLOAT', 'STRING', 'BOOL'] and 
            current_token.name in ['RBRACE', 'TYPE', 'IDENTIFIER', 'KEYWORD']):
            print(f"\n{green_text}Suggestion:{normal_text_start} Check if there should be a semicolon after '{prev_token.lexeme}'")
            
        # Mismatched type patterns
        elif (prev_token.name == 'TYPE' and current_token.name == 'ASSIGN_OP'):
            print(f"\n{green_text}Suggestion:{normal_text_start} Missing variable name after type '{prev_token.lexeme}'")
            
        # Missing closing parenthesis/brace
        elif prev_token.name == 'RPAREN' and current_token.name != 'LBRACE':
            print(f"\n{green_text}Suggestion:{normal_text_start} Expected opening brace '{{' after closing parenthesis")




    def generate_ir(self, parse_tree):
        """
        Transforms a parse tree into an Abstract Syntax Tree (AST) and Three-Address Code (TAC).

        Args:
            parse_tree: The root node of the parse tree

        Returns:
            tuple: (abstract_syntax_tree, three_address_code)
                - abstract_syntax_tree: A simplified, abstract representation of the program
                - three_address_code: A list of three-address instructions
        """
        # Initialize TAC instruction list and temporary counter
        tac = []
        temp_counter = 0
        label_counter = 0

        # Helper function to create new temporary variable names
        def new_temp():
            nonlocal temp_counter
            temp = f"t{temp_counter}"
            temp_counter += 1
            return temp

        # Helper function to create new label names
        def new_label():
            nonlocal label_counter
            label = f"L{label_counter}"
            label_counter += 1
            return label

        # The real AST node class - simpler than your current nodes
        class ASTNode_REAL:
            def __init__(self, kind, value=None, children=None):
                self.kind = kind  # The kind of node (e.g., 'program', 'function', 'if', 'binary_op')
                self.value = value  # Any associated value (e.g., operator type, variable name)
                self.children = children or []
                self.type = None  # For type information
                self.temp = None  # For storing temporary variable name

            def __str__(self):
                if self.value:
                    return f"{self.kind}({self.value})"
                return self.kind

        # Main recursive function to transform the parse tree
        def transform(node):
            if not node:
                return None, []

            # First, check for special node types: RelationalOperation and BinaryOperation
            if node.type == 'RelationalOperation' or node.type == 'BinaryOperation':
                if len(node.children) >= 2:
                    left_node, left_tac = transform(node.children[0])
                    right_node, right_tac = transform(node.children[1])

                    if left_node and right_node:
                        bin_op = ASTNode_REAL('binary_op', node.value)
                        bin_op.children = [left_node, right_node]
                        bin_op.temp = new_temp()

                        combined_tac = left_tac + right_tac
                        combined_tac.append(f"{bin_op.temp} = {left_node.temp} {node.value} {right_node.temp}")

                        return bin_op, combined_tac

            # Leaf nodes (terminals) - simplify them
            if not node.children:
                if node.type in ['IDENTIFIER', 'NUMBER', 'FLOAT', 'STRING', 'BOOL']:
                    ast_node = ASTNode_REAL('literal' if node.type != 'IDENTIFIER' else 'variable', node.value)
                    ast_node.temp = node.value if node.type == 'IDENTIFIER' else new_temp()

                    # Generate TAC for literals
                    tac_instructions = []
                    if node.type != 'IDENTIFIER':
                        tac_instructions.append(f"{ast_node.temp} = {node.value}")

                    return ast_node, tac_instructions
                return None, []

            # Process based on node type
            if node.type == 'Program':
                # The program consists of a single main function in your language
                child_ast, child_tac = transform(node.children[0])  # MainFunction
                prog_node = ASTNode_REAL('program', children=[child_ast])
                return prog_node, child_tac

            elif node.type == 'MainFunction':
                # Find the return type and the function body
                return_type = None
                body_ast = None
                body_tac = []

                # Identify type and body
                for child in node.children:
                    if child.type == 'Type':
                        for type_child in child.children:
                            if type_child.type == 'TYPE':
                                return_type = type_child.value
                    elif child.type == 'StatementList':
                        body_ast, body_tac = transform(child)

                # Create function node
                func_node = ASTNode_REAL('function', 'main', [])
                func_node.type = return_type

                if body_ast:
                    func_node.children.append(body_ast)

                # Add function entry in TAC
                tac_result = [f"function main:"] + body_tac
                return func_node, tac_result

            elif node.type == 'StatementList':
                # Combine all statements
                stmts_node = ASTNode_REAL('block')
                all_tac = []

                for child in node.children:
                    child_ast, child_tac = transform(child)
                    if child_ast:
                        stmts_node.children.append(child_ast)
                    all_tac.extend(child_tac)

                return stmts_node, all_tac

            elif node.type == 'Statement':
                # Pass through to the actual statement
                if node.children and len(node.children) > 0:
                    return transform(node.children[0])
                return None, []

            elif node.type == 'Declaration':
                # Variable declaration
                var_type = None
                var_name = None
                init_expr = None
                init_tac = []

                # Extract parts of the declaration
                for child in node.children:
                    if child.type == 'Type':
                        for type_child in child.children:
                            if type_child.type == 'TYPE':
                                var_type = type_child.value
                    elif child.type == 'IDENTIFIER':
                        var_name = child.value
                    elif child.type == 'Expression':
                        init_expr, init_tac = transform(child)

                # Create the declaration node
                decl_node = ASTNode_REAL('declaration', var_name)
                decl_node.type = var_type

                # Generate declaration and initialization TAC
                var_tac = []

                # Add initialization code first if present
                if init_expr:
                    decl_node.children.append(init_expr)
                    var_tac.extend(init_tac)

                # Add variable declaration
                var_tac.append(f"declare {var_type} {var_name}")

                # Add assignment if initialized
                if init_expr:
                    var_tac.append(f"{var_name} = {init_expr.temp}")

                return decl_node, var_tac

            elif node.type == 'ReturnStatement':
                # Find expression in children if present
                return_expr = None
                expr_tac = []

                for child in node.children:
                    if child.type == 'Expression':
                        return_expr, expr_tac = transform(child)
                        break
                    elif child.type == 'IDENTIFIER':
                        # Handle direct identifier as return value
                        return_expr = ASTNode_REAL('variable', child.value)
                        return_expr.temp = child.value
                        break
                    
                # Create return node
                return_node = ASTNode_REAL('return')

                if return_expr:
                    return_node.children.append(return_expr)
                    # Return with expression
                    return_tac = expr_tac + [f"return {return_expr.temp}"]
                else:
                    # Void return
                    return_tac = ["return"]

                return return_node, return_tac

            elif node.type == 'Assignment':
                # Assignment expression
                var_name = None
                expr = None
                expr_tac = []

                # Find variable name and expression
                for i, child in enumerate(node.children):
                    if child.type == 'IDENTIFIER' and i == 0:
                        var_name = child.value
                    elif child.type == 'Expression' or child.type == 'LogicalExpr' or child.type == 'ArithmeticExpr':
                        expr, expr_tac = transform(child)

                # Create assignment node
                assign_node = ASTNode_REAL('assignment', var_name)
                if expr:
                    assign_node.children.append(expr)
                    # Add the assignment instruction to TAC
                    expr_tac.append(f"{var_name} = {expr.temp}")

                return assign_node, expr_tac

            elif node.type == 'IfStatement':
                # If statement with optional else
                condition = None
                then_block = None
                else_block = None
                cond_tac = []
                then_tac = []
                else_tac = []

                # Find parts of the if statement
                for i, child in enumerate(node.children):
                    if child.type in ['Condition', 'Expression', 'LogicalExpr']:
                        condition, cond_tac = transform(child)
                    elif child.type == 'StatementList' and not then_block:
                        then_block, then_tac = transform(child)
                    elif child.type == 'StatementList':
                        else_block, else_tac = transform(child)

                # Create if node
                if_node = ASTNode_REAL('if')
                if condition:
                    if_node.children.append(condition)
                if then_block:
                    if_node.children.append(then_block)
                if else_block:
                    if_node.children.append(else_block)

                # Generate TAC for if statement with labels
                then_label = new_label()
                end_label = new_label()
                else_label = new_label() if else_block else end_label

                if_tac = cond_tac + [
                    f"if {condition.temp} goto {then_label}",
                    f"goto {else_label}"
                ]

                if_tac.append(f"{then_label}:")
                if_tac.extend(then_tac)
                if_tac.append(f"goto {end_label}")

                if else_block:
                    if_tac.append(f"{else_label}:")
                    if_tac.extend(else_tac)

                if_tac.append(f"{end_label}:")

                return if_node, if_tac

            elif node.type == 'WhileStatement':
                # While loop
                condition = None
                body = None
                cond_tac = []
                body_tac = []

                # Find condition and body
                for child in node.children:
                    if child.type in ['Condition', 'Expression', 'LogicalExpr']:
                        condition, cond_tac = transform(child)
                    elif child.type == 'StatementList':
                        body, body_tac = transform(child)

                # Create while node
                while_node = ASTNode_REAL('while')
                if condition:
                    while_node.children.append(condition)
                if body:
                    while_node.children.append(body)

                # Generate TAC for while statement
                start_label = new_label()
                body_label = new_label()
                end_label = new_label()

                while_tac = [
                    f"{start_label}:"
                ]
                while_tac.extend(cond_tac)
                while_tac.extend([
                    f"if {condition.temp} goto {body_label}",
                    f"goto {end_label}",
                    f"{body_label}:"
                ])
                while_tac.extend(body_tac)
                while_tac.append(f"goto {start_label}")
                while_tac.append(f"{end_label}:")

                return while_node, while_tac

            elif node.type == 'ForStatement':
                # For loop - init; condition; update; body
                init = None
                condition = None
                update = None
                body = None
                init_tac = []
                cond_tac = []
                update_tac = []
                body_tac = []

                # Find the components
                i = 0
                for child in node.children:
                    if i == 0 and child.type == 'Declaration':
                        init, init_tac = transform(child)
                        i += 1
                    elif i == 1 and child.type in ['Condition', 'Expression', 'LogicalExpr']:
                        condition, cond_tac = transform(child)
                        i += 1
                    elif i == 2 and child.type == 'Assignment':
                        update, update_tac = transform(child)
                        i += 1
                    elif child.type == 'StatementList':
                        body, body_tac = transform(child)

                # Create for node
                for_node = ASTNode_REAL('for')
                for child_node in [init, condition, update, body]:
                    if child_node:
                        for_node.children.append(child_node)

                # Generate TAC for for loop
                start_label = new_label()
                body_label = new_label()
                update_label = new_label()
                end_label = new_label()

                # Initial setup
                for_tac = init_tac + [
                    f"{start_label}:"
                ]

                # Condition check
                for_tac.extend(cond_tac)
                for_tac.extend([
                    f"if {condition.temp} goto {body_label}",
                    f"goto {end_label}",
                    f"{body_label}:"
                ])

                # Loop body
                for_tac.extend(body_tac)

                # Update expression
                for_tac.append(f"{update_label}:")
                for_tac.extend(update_tac)
                for_tac.append(f"goto {start_label}")
                for_tac.append(f"{end_label}:")

                return for_node, for_tac

            # Handle binary operations in expressions
            elif node.type in ['ArithmeticExpr', 'RelationalExpr', 'LogicalExpr'] and len(node.children) >= 3:
                # Binary operations
                left_node, left_tac = transform(node.children[0])
                op_node = node.children[1]
                right_node, right_tac = transform(node.children[2])

                if left_node and right_node and hasattr(op_node, 'value'):
                    # Create binary operation node
                    bin_op = ASTNode_REAL('binary_op', op_node.value)
                    bin_op.children = [left_node, right_node]

                    # Generate the temporary result
                    bin_op.temp = new_temp()

                    # Combine TAC and add operation
                    combined_tac = left_tac + right_tac
                    combined_tac.append(f"{bin_op.temp} = {left_node.temp} {op_node.value} {right_node.temp}")

                    return bin_op, combined_tac

            elif node.type == 'LogicalExpr' and node.children and node.children[0].type == 'not':
                # Unary not operator
                expr, expr_tac = transform(node.children[1])
                if not expr:
                    return None, []

                unary_op = ASTNode_REAL('unary_op', 'not')
                unary_op.children = [expr]

                # Generate temporary result
                unary_op.temp = new_temp()

                # Add the not operation to TAC
                unary_tac = expr_tac + [f"{unary_op.temp} = not {expr.temp}"]

                return unary_op, unary_tac

            elif node.type == 'Term' and len(node.children) == 3:
                # Handle multiplication, division, remainder
                left_node, left_tac = transform(node.children[0])
                op_node = node.children[1]
                right_node, right_tac = transform(node.children[2])

                if not (left_node and right_node):
                    return None, []

                # Create binary operation node
                bin_op = ASTNode_REAL('binary_op', op_node.value)
                bin_op.children = [left_node, right_node]

                # Generate temporary result
                bin_op.temp = new_temp()

                # Combine TAC and add operation
                combined_tac = left_tac + right_tac
                combined_tac.append(f"{bin_op.temp} = {left_node.temp} {op_node.value} {right_node.temp}")

                return bin_op, combined_tac

            elif node.type == 'PowerExpr' and len(node.children) == 3:
                # Handle exponentiation
                left_node, left_tac = transform(node.children[0])
                right_node, right_tac = transform(node.children[2])

                if not (left_node and right_node):
                    return None, []

                # Create power operation node
                power_op = ASTNode_REAL('binary_op', 'power')
                power_op.children = [left_node, right_node]

                # Generate temporary result
                power_op.temp = new_temp()

                # Combine TAC and add operation
                combined_tac = left_tac + right_tac
                combined_tac.append(f"{power_op.temp} = {left_node.temp} power {right_node.temp}")

                return power_op, combined_tac

            elif node.type == 'Factor' and node.children and node.children[0].type == '(':
                # Parenthesized expression
                return transform(node.children[1])

            # Pass through for wrapper node types with one child
            elif node.type in ['Expression', 'Condition', 'Factor', 'PowerExpr', 'Term'] and node.children:
                # Check if this is a wrapper around a special operation
                if len(node.children) == 1:
                    return transform(node.children[0])

                # Try to find binary operations
                for i, child in enumerate(node.children):
                    if i > 0 and i < len(node.children) - 1:
                        if child.type in ['ARITHMETIC_OP', 'RELATIONAL_OP', 'LOGICAL_OP']:
                            left_node, left_tac = transform(node.children[i-1])
                            right_node, right_tac = transform(node.children[i+1])

                            if left_node and right_node:
                                bin_op = ASTNode_REAL('binary_op', child.value)
                                bin_op.children = [left_node, right_node]
                                bin_op.temp = new_temp()

                                combined_tac = left_tac + right_tac
                                combined_tac.append(f"{bin_op.temp} = {left_node.temp} {child.value} {right_node.temp}")

                                return bin_op, combined_tac

                # Default behavior if no special handling needed
                return transform(node.children[0])

            # For any node type we didn't handle specifically
            for child in node.children:
                # Try to check for special nodes (RelationalOperation, BinaryOperation)
                if child.type in ['RelationalOperation', 'BinaryOperation']:
                    child_ast, child_tac = transform(child)
                    if child_ast:
                        return child_ast, child_tac

            # Last resort - check each child for results
            for child in node.children:
                child_ast, child_tac = transform(child)
                if child_ast:
                    return child_ast, child_tac

            return None, []

        # Start the transformation from the root
        ast_root, tac_instructions = transform(parse_tree)

        return ast_root, tac_instructions



    def visualize_ast_real(self, ast_node, level=0):
        """
        Visualize the actual AST with proper indentation.
        """
        indent = "  " * level
        if ast_node.value:
            print(f"{indent}{ast_node.kind}: {ast_node.value}")
        else:
            print(f"{indent}{ast_node.kind}")
            
        for child in ast_node.children:
            self.visualize_ast_real(child, level + 1)
            
    def display_tac(self, tac_instructions):
        """
        Display the three-address code instructions.
        """
        print("\n=== Three-Address Code ===")
        for i, instruction in enumerate(tac_instructions):
            if instruction.endswith(':'):
                print(f"\n{instruction}")
            else:
                print(f"    {i}: {instruction}")