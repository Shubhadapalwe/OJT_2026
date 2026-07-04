# =========================================================
#              Marvellous Infosystems
# =========================================================
# Project Name : Marvellous Infosystems Calculator
# Author       : Piyush Manohar Khairnar
# Date         : 09/05/2026
# =========================================================

import tkinter as tk
from tkinter import messagebox


# =========================================================
# Function : click_button
# =========================================================

def click_button(value):

    current = entry.get()

    entry.delete(0, tk.END)

    entry.insert(0, current + str(value))


# =========================================================
# Function : clear_screen
# =========================================================

def clear_screen():

    entry.delete(0, tk.END)


# =========================================================
# Function : calculate_result
# =========================================================

def calculate_result():

    try:

        expression = entry.get()

        result = eval(expression)

        entry.delete(0, tk.END)

        entry.insert(0, str(result))

    except:

        messagebox.showerror(
            "Error",
            "Invalid Expression"
        )

        entry.delete(0, tk.END)


# ---------------- Main Window ----------------

root = tk.Tk()

root.title("Marvellous Infosystems Calculator")

root.geometry("400x600")

root.configure(bg="#0f172a")


# ---------------- Title ----------------

title = tk.Label(
    root,
    text="Marvellous Infosystems",
    font=("Arial", 22, "bold"),
    bg="#0f172a",
    fg="#38bdf8"
)

title.pack(pady=10)


# ---------------- Tagline ----------------

subtitle = tk.Label(
    root,
    text="आम्ही Technical संस्कार करतो !!!",
    font=("Arial", 12, "bold"),
    bg="#0f172a",
    fg="#facc15"
)

subtitle.pack(pady=5)


# ---------------- Entry Box ----------------

entry = tk.Entry(
    root,
    font=("Arial", 24, "bold"),
    width=18,
    bd=5,
    relief="ridge",
    justify="right",
    bg="white",
    fg="black"
)

entry.pack(pady=20)


# ---------------- Button Frame ----------------

button_frame = tk.Frame(
    root,
    bg="#0f172a"
)

button_frame.pack()


# ---------------- Buttons ----------------

buttons = [
    ["7", "8", "9", "/"],
    ["4", "5", "6", "*"],
    ["1", "2", "3", "-"],
    ["0", ".", "=", "+"]
]


# ---------------- Create Buttons ----------------

for row in buttons:

    row_frame = tk.Frame(
        button_frame,
        bg="#0f172a"
    )

    row_frame.pack()

    for button in row:

        # Operator Buttons
        if button in ["+", "-", "*", "/"]:

            btn = tk.Button(
                row_frame,
                text=button,
                font=("Arial", 18, "bold"),
                width=5,
                height=2,
                bg="#60a5fa",
                fg="black",
                activebackground="#3b82f6",
                activeforeground="black",
                relief="raised",
                bd=3,
                command=lambda x=button: click_button(x)
            )

        # Equal Button
        elif button == "=":

            btn = tk.Button(
                row_frame,
                text=button,
                font=("Arial", 18, "bold"),
                width=5,
                height=2,
                bg="#4ade80",
                fg="black",
                activebackground="#22c55e",
                activeforeground="black",
                relief="raised",
                bd=3,
                command=calculate_result
            )

        # Number Buttons
        else:

            btn = tk.Button(
                row_frame,
                text=button,
                font=("Arial", 18, "bold"),
                width=5,
                height=2,
                bg="#e2e8f0",
                fg="black",
                activebackground="#cbd5e1",
                activeforeground="black",
                relief="raised",
                bd=3,
                command=lambda x=button: click_button(x)
            )

        btn.pack(
            side="left",
            padx=5,
            pady=5
        )


# ---------------- Clear Button ----------------

clear_btn = tk.Button(
    root,
    text="CLEAR",
    font=("Arial", 16, "bold"),
    width=24,
    height=2,
    bg="#f87171",
    fg="black",
    activebackground="#ef4444",
    activeforeground="black",
    relief="raised",
    bd=4,
    command=clear_screen
)

clear_btn.pack(pady=20)


# ---------------- Footer ----------------

footer = tk.Label(
    root,
    text="Designed by Marvellous Infosystems",
    font=("Arial", 10, "bold"),
    bg="#0f172a",
    fg="#94a3b8"
)

footer.pack(pady=10)


# ---------------- Main Loop ----------------

root.mainloop()