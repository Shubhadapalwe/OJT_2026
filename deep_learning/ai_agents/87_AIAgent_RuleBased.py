print("Marvellous Infosystems")
print("Simple Rule Based AI Agent")

def simple_agent(user_input):
    user_input = user_input.lower()

    if "hello" in user_input:
        return "Hello Jay Ganesh.. How can I help you?"

    elif "course" in user_input:
        return "We provide courses in C, C++, Java, Python, AI, ML and Logic Building."

    elif "fees" in user_input:
        return "Please contact 7020713938 for detailed fee structure."

    elif "bye" in user_input:
        return "Thank you! Have a great day."

    else:
        return "Sorry, I did not understand your query."

while True:
    query = input("User: ")

    response = simple_agent(query)
    print("Agent:", response)

    if "bye" in query.lower():
        break