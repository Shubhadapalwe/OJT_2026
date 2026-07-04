# AI Agents — From Rule-Based to Memory-Enabled

**Deep Learning & AI Portfolio**
**Trainee: Shubhada A. Palwe**

---

## What's in This Folder

| File | Agent Type | What it does |
|------|-----------|-------------|
| `87_AIAgent_RuleBased.py` | Rule-based | Responds to fixed keywords (hello, course, fees, bye) |
| `88_AIAGentCalculator.py` | Tool-using | Detects "calculate" keyword, evaluates math |
| `89_AIAGenyMultpleTool.py` | Multi-tool | Routes to 4 tools: calculator, time, date, courses |
| `90_AIAgent_Memory.py` | Memory-enabled | Remembers user's name across the conversation |

---

## What is an AI Agent?

An AI agent is a program that:
1. Takes user input
2. Decides what action to take (routing / reasoning)
3. Executes a tool or returns a response
4. Loops back and waits for the next input

These four files show a progression — each one adds a new capability on top of the previous idea. Together they cover the core patterns you'll see in even the most advanced agent frameworks (LangChain, AutoGen, etc.).

---

## File 87 — Rule-Based Agent (`87_AIAgent_RuleBased.py`)

### The Code

```python
def simple_agent(user_input):
    user_input = user_input.lower()

    if "hello" in user_input:
        return "Hello! Welcome to Marvellous Infosystems."
    elif "course" in user_input:
        return "We offer Python, ML, DL, and AI courses."
    elif "fees" in user_input:
        return "Our course fee is Rs. 15,000."
    elif "bye" in user_input:
        return "Goodbye! Have a great day."
    else:
        return "I'm sorry, I don't understand that."

while True:
    user_input = input("User: ")
    if "bye" in user_input.lower():
        print("Agent:", simple_agent(user_input))
        break
    print("Agent:", simple_agent(user_input))
```

### How it Works

This is the simplest possible agent: pure `if/elif` routing. The agent has no intelligence beyond pattern matching — it checks if a keyword appears anywhere in the lowercased input and returns a fixed response.

- `user_input.lower()` makes matching case-insensitive ("Hello" and "HELLO" both match)
- The `while True` loop keeps the conversation going until the user types "bye"
- No external libraries needed — just Python

**When would you use this?** Simple FAQ chatbots, customer service bots with a fixed set of questions, menu navigation systems. If you know exactly what users will ask, rule-based agents are fast, predictable, and easy to maintain.

---

## File 88 — Calculator Agent (`88_AIAGentCalculator.py`)

### The Code

```python
def calculator(expression):
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

def agent(user_input):
    if "calculate" in user_input.lower():
        expression = user_input.lower().replace("calculate", "").strip()
        return calculator(expression)
    else:
        return "I can only calculate. Try: calculate 5 + 3 * 2"

while True:
    query = input("User: ")
    if query.lower() == "exit":
        print("Agent: Goodbye!")
        break
    print("Agent:", agent(query))
```

### How it Works

This agent introduces the concept of **tool use**. The agent has two parts:

1. **The router** (`agent()`) — decides *what to do* with the input
2. **The tool** (`calculator()`) — actually *does the thing*

When the user types "calculate 10 * 5 + 3":
- Router detects "calculate" in the input
- Strips the word "calculate" out: `"10 * 5 + 3"`
- Passes the remaining expression to the calculator tool
- `eval("10 * 5 + 3")` → `53`
- Returns "Result: 53"

**Note:** `eval()` is convenient here because it handles any valid Python math expression, including operator precedence (`*` before `+`). Same caveat as the tkinter calculator — fine for learning, but use a safe parser in production.

This pattern — route to tool, execute, return result — is exactly what LangChain and other agent frameworks do under the hood, just with much more sophisticated routing (using an LLM instead of keyword matching).

---

## File 89 — Multi-Tool Agent (`89_AIAGenyMultpleTool.py`)

### The Code

```python
import datetime

def calculator(expression):
    try:
        result = eval(expression)
        return f"Result: {result}"
    except:
        return "Invalid expression."

def show_time():
    return datetime.datetime.now().strftime("%H:%M:%S")

def show_date():
    return datetime.datetime.now().strftime("%d-%m-%Y")

def show_courses():
    return "We offer: Python, Machine Learning, Deep Learning, AI Agents"

def agent(user_input):
    user_input_lower = user_input.lower()

    if "calculate" in user_input_lower:
        expression = user_input_lower.replace("calculate", "").strip()
        return calculator(expression)
    elif "time" in user_input_lower:
        return show_time()
    elif "date" in user_input_lower:
        return show_date()
    elif "course" in user_input_lower:
        return show_courses()
    else:
        return "I can help with: calculate, time, date, courses"

while True:
    query = input("User: ")
    if query.lower() == "exit":
        print("Goodbye!")
        break
    print("Agent:", agent(query))
```

### How it Works

Now the agent has 4 tools to choose from. The router checks for different keywords and dispatches to the right function:

| User says | Keyword detected | Tool called |
|-----------|-----------------|-------------|
| "calculate 5+3" | "calculate" | `calculator()` |
| "what is the time" | "time" | `show_time()` |
| "show today's date" | "date" | `show_date()` |
| "what courses do you offer" | "course" | `show_courses()` |

The `datetime` module gives us real-time values:
- `strftime("%H:%M:%S")` → "14:35:22" (24-hour clock: hour, minute, second)
- `strftime("%d-%m-%Y")` → "04-07-2026" (day-month-year)

**The key lesson here:** agents are essentially just routers. The more tools they have access to, and the smarter the routing logic, the more powerful they become. Replace the keyword-matching router with an LLM call and you get something like a GPT function-calling agent.

---

## File 90 — Memory-Enabled Agent (`90_AIAgent_Memory.py`)

### The Code

```python
memory = {}

def agent(user_input):
    if "my name is" in user_input.lower():
        name = user_input.lower().replace("my name is", "").strip()
        memory["name"] = name
        return f"Nice to meet you, {name}"

    elif "what is my name" in user_input.lower():
        if "name" in memory:
            return f"Your name is {memory['name']}"
        else:
            return "I do not know your name yet."

    elif "show memory" in user_input.lower():
        return str(memory)

    else:
        return "I can remember your name. Try: My name is Rahul"

while True:
    query = input("User: ")
    if query.lower() == "exit":
        print("Agent: Goodbye!")
        break
    print("Agent:", agent(query))
```

### How it Works

This agent introduces **memory** — the ability to remember information across turns in a conversation. The memory is just a Python dictionary:

```python
memory = {}
```

It persists for as long as the program is running. When the user says "My name is Rahul":
- The agent extracts "rahul" (lowercased) from the input
- Stores it: `memory["name"] = "rahul"`
- Returns a greeting

Later, when the user asks "What is my name?":
- The agent checks if "name" exists in `memory`
- If yes: returns it
- If no: says it doesn't know yet

The user can also type "show memory" to see the full contents of the dictionary — useful for debugging.

**Why does this matter?** Standard LLM chatbots without memory start fresh with every message. They don't know what you told them 3 messages ago unless the conversation history is explicitly passed back in. This simple dictionary is a minimal version of what "conversation memory" means in agent frameworks — storing key facts extracted from earlier turns.

### What `memory` looks like after a conversation

```
User: My name is Priya
Agent: Nice to meet you, priya

User: show memory
Agent: {'name': 'priya'}

User: What is my name?
Agent: Your name is priya
```

---

## The Agent Progression

Looking at all four files together, each one adds one new concept:

```
File 87: Rule-based routing only
           ↓
File 88: Rule-based routing + one tool (calculator)
           ↓
File 89: Rule-based routing + multiple tools (calculator, time, date, courses)
           ↓
File 90: Rule-based routing + memory (persistent state across turns)
```

A production agent (like a GPT-4 agent with tools) combines all of these:
- LLM-based routing (instead of keyword matching)
- Many tools (search, code execution, APIs, databases)
- Long-term memory (vector stores, databases)
- Multi-step planning (chain of tool calls)

But the fundamental ideas — route, execute, remember — are exactly what we've built here.

---

## How to Run

All four agents run in the terminal:

```bash
python 87_AIAgent_RuleBased.py
python 88_AIAGentCalculator.py
python 89_AIAGenyMultpleTool.py
python 90_AIAgent_Memory.py
```

For Files 88/89, try inputs like:
- `calculate 100 / 4 + 7`
- `what time is it`
- `show date`
- `what courses do you have`

For File 90:
- `My name is Rahul`
- `What is my name?`
- `show memory`
- `exit`

**Note:** No external libraries needed for any of these four files — just Python's built-in `datetime` for File 89.

---

## Key Concepts Summary

| Concept | File | Explanation |
|---------|------|-------------|
| Keyword routing | 87 | `if/elif` on lowercased input |
| Tool use | 88 | Separate function called by router |
| Multi-tool routing | 89 | Router chooses from 4 tools |
| eval() for math | 88, 89 | Evaluates string as Python expression |
| strftime() | 89 | Format datetime as human-readable string |
| Agent memory | 90 | Dictionary persists across conversation turns |
| Graceful exit | All | Loop breaks on "bye" / "exit" |
