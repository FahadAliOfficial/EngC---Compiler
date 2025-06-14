# EngC - English-like Programming Language Compiler

A complete compiler implementation for a custom English-like programming language that translates human-readable code into optimized assembly and C++ code. This project demonstrates the full compiler pipeline from lexical analysis to code optimization and generation.

## 📋 Table of Contents
- [Overview](#overview)
- [Language Features](#language-features)
- [Project Structure](#project-structure)
- [Compiler Pipeline](#compiler-pipeline)
- [Installation & Setup](#installation--setup)
- [Usage Examples](#usage-examples)
- [Technical Implementation](#technical-implementation)
- [Testing](#testing)
- [Contributors](#contributors)

## 🌟 Overview

EngC is an educational compiler project that implements a complete compilation pipeline for an English-like programming language. The language uses natural English phrases for operators and control structures, making code more readable and intuitive for beginners.

### Example Code
```english
integer main () {
    integer x equals to 5 semicolon
    integer y equals to 10 semicolon
    boolean result equals to x is less than y semicolon
    if (result) {
        integer sum equals to x add y semicolon
    }
}
```

## 🔤 Language Features

### Data Types
- `integer` - Whole numbers
- `float` - Decimal numbers  
- `string` - Text values
- `boolean` - true/false values

### Operators
- **Arithmetic**: `add`, `subtract`, `multiply`, `divide`, `remainder`, `power`
- **Relational**: `is equal to`, `is not equal to`, `is greater than`, `is less than`, `is greater than or equal to`, `is less than or equal to`
- **Logical**: `and`, `or`, `not`
- **Assignment**: `equals to`, `add equals to`, `subtract equals to`, etc.

### Control Structures
- **Conditional**: `if-else` statements
- **Loops**: `while` and `for` loops
- **Functions**: `main` function support

### Language Grammar
- Natural English syntax
- Semicolon statement terminators
- Braces for code blocks
- Parentheses for expressions and conditions

## 📁 Project Structure

```
EngC---Compiler/
├── Lexer_Hadeed/              # Lexical Analysis Component
│   ├── lexer.ipynb           # Main lexer implementation
│   ├── scanner.py            # Token scanner with DFA table
│   ├── Token_class.py        # Token data structure
│   ├── token_definition.py   # Token type definitions
│   ├── tokenize_regex.py     # Regular expression tokenizer
│   ├── postfix_conversion.py # Regex to postfix conversion
│   ├── postfix_to_nfa.py     # NFA construction from postfix
│   ├── nfa_to_dfa.py         # NFA to DFA conversion
│   ├── optimize_dfa.py       # DFA optimization algorithms
│   ├── dfa_table.py          # DFA transition table generator
│   ├── plot_nfa.py           # NFA/DFA visualization
│   └── language.txt          # Language token specifications
├── Parser-Junaid/             # Syntax Analysis Component
│   ├── LR1Parser.py          # LR(1) parser implementation
│   ├── tester.ipynb          # Parser testing notebook
│   └── combined_parsing_table.xlsx # Parsing table export
├── Optimizer-Assembly-CPP-Fahad/ # Code Optimization & Generation Component
│   ├── Optimize_assembly_cpp.ipynb # TAC to Assembly & C++ generation
│   └── __pycache__/          # Compiled Python modules
└── README.md                 # This file
```

## 🔄 Compiler Pipeline

### 1. Lexical Analysis (Lexer_Hadeed)
- **Input**: Source code text
- **Process**: 
  - Tokenizes input using DFA-based scanner
  - Converts regular expressions to NFA, then optimized DFA
  - Generates transition tables for efficient token recognition
- **Output**: Stream of tokens with types and lexemes

### 2. Syntax Analysis (Parser-Junaid)
- **Input**: Token stream from lexer
- **Process**:
  - LR(1) parsing with conflict resolution
  - Grammar-driven parse tree construction
  - Semantic analysis (type checking, variable declaration)
- **Output**: Parse tree and semantic validation

### 3. Intermediate Representation
- **Parse Tree → AST**: Simplified abstract syntax tree
- **TAC Generation**: Three-address code instructions
- **Optimizations**: Control flow and expression optimization

### 4. Code Optimization & Generation (Optimizer-Assembly-CPP-Fahad)
- **Input**: Optimized Three-address code (TAC)
- **Process**:
  - TAC optimization and analysis
  - Register allocation strategies
  - Instruction scheduling
  - Dead code elimination
  - Dual code generation pipeline
- **Output**: 
  - Optimized assembly code
  - Equivalent C++ code

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook
Required Python packages:
- pandas
- matplotlib
- numpy
- copy
- re
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd EngC---Compiler

# Install dependencies
pip install pandas matplotlib numpy jupyter

# Launch Jupyter Notebook
jupyter notebook
```

## 💡 Usage Examples

### Running the Complete Pipeline

1. **Lexical Analysis**:
```python
# In Lexer_Hadeed/lexer.ipynb
from scanner import LexicalScanner
from dfa_table import DFATable

# Load DFA table and create scanner
dfa_table = DFATable()
scanner = LexicalScanner(dfa_table)

# Tokenize source code
source_code = 'integer x equals to 5 semicolon'
tokens = scanner.scan(source_code)
```

2. **Parsing & Semantic Analysis**:
```python
# In Parser-Junaid/tester.ipynb
from LR1Parser import LR1Parser

parser = LR1Parser()
ast, tac = parser.test_program(
    tokens, 
    "simple declaration", 
    "integer x equals to 5;",
    create_visualization=True
)
```

3. **Code Generation & Optimization**:
```python
# In Optimizer-Assembly-CPP-Fahad/Optimize_assembly_cpp.ipynb
from aggressive_optimizer import complete_compiler_pipeline

# Generate both assembly and C++ from optimized TAC
assembly_code, cpp_code = complete_compiler_pipeline(
    'Test Program', 
    tac_code, 
    expected_result
)
```

### Sample Programs

**Variable Declaration and Assignment**:
```english
integer main () {
    integer age equals to 25 semicolon
    float height equals to 5.8 semicolon
    string name equals to "John" semicolon
    boolean isStudent equals to true semicolon
}
```

**Conditional Logic**:
```english
integer main () {
    integer x equals to 10 semicolon
    integer y equals to 20 semicolon
    if (x is less than y) {
        integer result equals to x add y semicolon
    } else {
        integer result equals to x subtract y semicolon
    }
}
```

**Loops**:
```english
integer main () {
    integer i equals to 0 semicolon
    while (i is less than 10) {
        i add equals to 1 semicolon
    }
    
    for (integer j equals to 0 semicolon j is less than 5 semicolon j add equals to 1) {
        integer temp equals to j multiply 2 semicolon
    }
}
```

## 🔧 Technical Implementation

### Lexer Features
- **DFA-based Recognition**: Efficient finite automaton for token recognition
- **Regular Expression Support**: Flexible token pattern definitions
- **Optimization**: Unreachable state removal and state minimization
- **Visualization**: Graphical NFA/DFA representation

### Parser Features
- **LR(1) Parsing**: Bottom-up parsing with 1-token lookahead
- **Conflict Resolution**: Precedence rules for operator conflicts
- **Error Recovery**: Helpful error messages with context
- **AST Generation**: Clean abstract syntax tree construction
- **Semantic Analysis**: Type checking and variable validation

### Intermediate Representation
- **Three-Address Code**: Linear intermediate representation
- **Control Flow**: Proper label generation for jumps and loops
- **Type Information**: Maintained throughout compilation
- **Optimization Ready**: Structure suitable for optimization passes

### Code Generation & Optimization
- **Dual Target Support**: Generates both assembly and C++ from optimized TAC
- **Register Allocation**: Efficient register usage strategies
- **Instruction Selection**: Optimal assembly instruction mapping
- **Optimization Passes**: Multiple optimization strategies
- **Assembly Output**: Human-readable assembly code
- **C++ Output**: High-level C++ equivalent code

## 🧪 Testing

The project includes comprehensive test suites:

### Parser Tests (Parser-Junaid/tester.ipynb)
- ✅ Simple declarations and assignments
- ✅ Type compatibility checking
- ✅ Conditional statements (if-else)
- ✅ Loop constructs (for, while)
- ✅ Expression evaluation
- ✅ Error detection and reporting

### Test Categories
1. **Syntax Validation**: Correct parsing of valid programs
2. **Semantic Analysis**: Type checking and variable validation
3. **Error Handling**: Proper error detection and messaging
4. **Code Generation**: TAC and assembly output verification

### Running Tests
```python
# Run all parser tests
jupyter notebook Parser-Junaid/tester.ipynb

# Run lexer tests  
jupyter notebook Lexer_Hadeed/lexer.ipynb

# Run optimization tests
jupyter notebook Optimizer-Assembly-CPP-Fahad/Optimize_assembly_cpp.ipynb
```

## 👥 Contributors

This project was developed as a collaborative effort with specialized components:

- **Lexer_Hadeed**: Lexical analysis and finite automaton implementation
- **Parser-Junaid**: LR(1) parser and semantic analysis
- **Optimizer-Assembly-CPP-Fahad**: Code optimization, assembly generation, and C++ generation 

## 📊 Key Achievements

- ✅ **Complete Compiler Pipeline**: From source code to optimized assembly and C++ code
- ✅ **English-like Syntax**: Natural language programming constructs
- ✅ **Robust Error Handling**: Comprehensive error detection and reporting
- ✅ **Visualization Tools**: Parse tree and AST visualization
- ✅ **Dual Code Generation**: Both assembly and C++ output from TAC
- ✅ **Optimization Support**: Multiple optimization strategies
- ✅ **Educational Value**: Clear documentation and modular design

## 🔮 Future Enhancements

- Function definitions and calls
- Array and pointer support
- Advanced optimization techniques
- Code generation for multiple target architectures
- IDE integration with syntax highlighting
- Debugger support

---

**Note**: This is an educational compiler project designed to demonstrate compiler construction principles and techniques. The English-like syntax makes it an excellent learning tool for understanding how programming languages are processed and translated.