from transformers import pipeline

# Load the text generation model
pipe = pipeline("text-generation", model="mistralai/Mixtral-8x7B-v0.1")

# Sample text (you can replace this with your input)
texts = [
    "The recent elections in the United States have led to a change in policies.",
    "New advancements in quantum computing could revolutionize technology.",
    "Climate change is impacting ecosystems worldwide."
]

# Generate text for each input
for text in texts:
    result = pipe(text, max_length=50)  # You can adjust max_length as needed
    print(f"Input: {text}")
    print(f"Generated Text: {result[0]['generated_text']}")
    print("-----------")
