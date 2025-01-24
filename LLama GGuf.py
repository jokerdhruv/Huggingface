from llama_cpp import Llama
import numpy as np
import librosa
# Load the model
model = Llama(
    model_path="llama-2-7b.Q2_K.gguf",
    n_ctx=2048,
    n_gpu_layers=0  # For CPU
)

# Generate text
response = model.create_completion(
    prompt="What is regression?",
    max_tokens=100,  # Limit response length
    stop=[],  # Optional stop sequences
    echo=True  # Includes original prompt in output
)

print(response['choices'][0]['text'])