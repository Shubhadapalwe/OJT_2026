# tkinter Projects — Desktop GUI with Python

**Marvellous Infosystems | Deep Learning & AI Portfolio**
**Trainee: Shubhada A. Palwe**

---

## What's in This Folder

| File | What it does |
|------|-------------|
| `85_Marvellous_Calculator_tkinter.py` | Dark-themed desktop calculator built with Python's tkinter |

---

## Why tkinter?

Most of our projects so far have been either terminal programs or web apps (Streamlit). But sometimes you need a proper desktop GUI — a real window with buttons and text fields that runs as a standalone application. Python's `tkinter` library lets us do that without installing anything extra. It ships with Python itself.

We built a full calculator as our first tkinter project. It looks polished (dark theme, colored buttons) and handles real arithmetic using Python's `eval()` function.

---

## File 85 — Dark-Themed Calculator (`85_Marvellous_Calculator_tkinter.py`)

### The Overall Idea

A working calculator with:
- A dark UI (background `#0f172a`, almost black)
- A grid of buttons (numbers, operators, CLEAR, =)
- An input/display field at the top
- Error handling if the expression is invalid

### Setting Up the Window

```python
import tkinter as tk
from tkinter import messagebox

root = tk.Tk()
root.title("Marvellous Calculator")
root.geometry("400x600")
root.configure(bg="#0f172a")
```

- `tk.Tk()` creates the main application window
- `root.geometry("400x600")` sets width × height in pixels
- `root.configure(bg=...)` sets the window background color
- Everything we add to the calculator goes inside `root`

### The Display Field

```python
entry = tk.Entry(root, font=("Arial", 24), bg="#1e293b", fg="white",
                 justify="right", bd=0, insertbackground="white")
entry.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=10, pady=10)
```

- `tk.Entry` is a single-line text input — we use it as the calculator's display
- `columnspan=4` makes it stretch across all 4 button columns
- `sticky="nsew"` makes it fill the entire cell (north-south-east-west)
- `justify="right"` so numbers align right like a real calculator

### Handling Button Clicks

```python
def click_button(value):
    entry.insert(tk.END, value)
```

When a number or operator button is clicked, we just append its value to the entry field. `tk.END` means "insert at the end of whatever's already there."

```python
def calculate_result():
    expression = entry.get()
    try:
        result = eval(expression)
        entry.delete(0, tk.END)
        entry.insert(tk.END, str(result))
    except Exception:
        messagebox.showerror("Error", "Invalid Expression")
        entry.delete(0, tk.END)
```

- `entry.get()` reads whatever is currently in the display
- `eval()` evaluates the math expression (e.g. `"3+5*2"` → `13`)
- If `eval()` throws an error (e.g. the user typed `"3++"`), we show a popup with `messagebox.showerror` and clear the display

**Note:** `eval()` is convenient for a learning project like this. In production apps, you'd use a proper expression parser instead, because `eval()` can execute arbitrary Python code. For a local calculator, though, it's perfectly fine.

```python
def clear_entry():
    entry.delete(0, tk.END)
```

`entry.delete(0, tk.END)` clears from position 0 to the end — effectively erases everything.

### The Button Grid

We define all buttons in a 2D list:

```python
buttons = [
    ["7", "8", "9", "/"],
    ["4", "5", "6", "*"],
    ["1", "2", "3", "-"],
    ["0", ".", "=", "+"],
    ["CLEAR", "", "", ""],
]
```

Then we loop through and create each button:

```python
for row_index, row in enumerate(buttons):
    for col_index, button_text in enumerate(row):
        if button_text == "":
            continue

        if button_text == "CLEAR":
            btn = tk.Button(root, text=button_text, font=("Arial", 16, "bold"),
                            bg="#f87171", fg="white", command=clear_entry)
            btn.grid(row=row_index+1, column=col_index, columnspan=4,
                     sticky="nsew", padx=5, pady=5)
        elif button_text == "=":
            btn = tk.Button(root, text=button_text, font=("Arial", 16, "bold"),
                            bg="#22c55e", fg="white", command=calculate_result)
            btn.grid(row=row_index+1, column=col_index, sticky="nsew", padx=5, pady=5)
        else:
            btn = tk.Button(root, text=button_text, font=("Arial", 16, "bold"),
                            bg="#334155", fg="white",
                            command=lambda x=button_text: click_button(x))
            btn.grid(row=row_index+1, column=col_index, sticky="nsew", padx=5, pady=5)
```

**The `lambda x=button_text` trick is important.** If we just wrote `lambda: click_button(button_text)`, all buttons would end up clicking with the *last* value of `button_text` from the loop (because Python closures capture variables by reference, not by value). By writing `lambda x=button_text:`, we capture the current value as a default argument — so each button remembers its own label correctly.

- CLEAR button → red (`#f87171`), spans all 4 columns
- = button → green (`#22c55e`)
- All other buttons → dark slate (`#334155`)

### Starting the App

```python
root.mainloop()
```

This line starts the tkinter event loop. The window stays open and responsive to clicks until the user closes it. Everything in tkinter lives inside this loop — without it, the window would open and immediately close.

### Layout System: Grid

tkinter has three layout managers: `pack`, `place`, and `grid`. We use `grid` here because the calculator is literally a grid of buttons. Every widget gets a `row` and `column` position, and you can use `columnspan` to merge cells.

```
row=0: [Entry field spanning all 4 columns]
row=1: [7] [8] [9] [/]
row=2: [4] [5] [6] [*]
row=3: [1] [2] [3] [-]
row=4: [0] [.] [=] [+]
row=5: [CLEAR spanning all 4 columns]
```

---

## How to Run

```bash
python 85_Marvellous_Calculator_tkinter.py
```

**Note:** Unlike Streamlit apps, you run tkinter with plain `python`, not `streamlit run`. A window will pop up on your desktop. If you're on a headless server (no display), you'll get a `_tkinter.TclError: no display` error — tkinter needs a real screen.

---

## Key tkinter Concepts Summary

| Concept | What it does |
|---------|-------------|
| `tk.Tk()` | Creates the main window |
| `root.geometry("WxH")` | Sets window size in pixels |
| `tk.Entry` | Single-line text input / display |
| `tk.Button` | Clickable button |
| `widget.grid(row, col)` | Places widget in grid layout |
| `columnspan=N` | Widget spans N columns |
| `sticky="nsew"` | Widget fills its cell |
| `entry.get()` | Read current text |
| `entry.insert(tk.END, val)` | Append text to end |
| `entry.delete(0, tk.END)` | Clear all text |
| `root.mainloop()` | Start the event loop (keep window open) |
| `lambda x=val: f(x)` | Capture loop variable correctly |

---

## What We Learned

tkinter gives us a quick way to build desktop GUIs in pure Python. The key ideas are:

1. Create a root window with `tk.Tk()`
2. Add widgets (Entry, Button, Label, etc.) and arrange them with `.grid()`
3. Connect user actions to Python functions via the `command=` parameter
4. Use `lambda x=val: ...` when creating buttons in a loop — otherwise all buttons share the same variable reference
5. Call `root.mainloop()` at the very end to start the event loop

This calculator is a solid foundation for any desktop tool — swap out the button layout and logic, and you can build forms, dashboards, or data entry tools the same way.
