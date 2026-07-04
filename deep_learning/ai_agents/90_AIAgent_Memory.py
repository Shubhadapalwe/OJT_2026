print("Marvellous Infosystems")
print("AI Agent with Memory")

memory = {}

def agent(user_input):
    words = user_input.split()

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