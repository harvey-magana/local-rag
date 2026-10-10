from ollama import Client
import time

client = Client(
    host="http://localhost:11434",
    timeout=600
)

start = time.time()

response = client.chat(
    model="llama3.1:latest",
    messages=[
        {
            "role": "user",
            "content": "Explain REST APIs in two short sentences."
        }
    ]
)

elapsed = time.time() - start

print(response["message"]["content"])
print(f"\nElapsed time: {elapsed:.2f} seconds")