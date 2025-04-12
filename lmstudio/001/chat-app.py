import lmstudio as lms

# http://host.docker.internal:1234"
llm = lms.llm()  # Get any loaded LLM
# prediction = llm.respond_stream("What is a Capybara?")
prediction = llm.respond_stream("2 + 2")

for token in prediction:
    print(token, end="", flush=True)

