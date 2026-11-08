import ollama

# Step 1: Define model
model_name = "llama3.2:1b"

# Step 2: Enter question
question = "What is software testing?"

# Step 3: Send question to Ollama
response = ollama.chat(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": question
        }
    ]
)

# Step 4: Display AI response
print("\nQuestion:", question)
print("\nOllama Response:")
print(response["message"]["content"])