print("Marvellous Infosystems")
print("AI Agent with Multiple Tools")

def calculator(expression):
    try:
        return eval(expression)
    except:
        return "Invalid expression"

def show_time():
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")

def show_date():
    import datetime
    return datetime.datetime.now().strftime("%d-%m-%Y")

def agent(user_input):
    user_input = user_input.lower()

    if "calculate" in user_input:
        expression = user_input.replace("calculate", "")
        return "Answer is: " + str(calculator(expression))

    elif "time" in user_input:
        return "Current time is: " + show_time()

    elif "date" in user_input:
        return "Today's date is: " + show_date()

    elif "course" in user_input:
        return "We teach C, C++, Java, Python, AI, ML, DL and GenAI."

    else:
        return "I am unable to understand. Please try again."

while True:
    query = input("User: ")

    if query.lower() == "exit":
        print("Agent: Thank you!")
        break

    print("Agent:", agent(query))