

import transformers
import numpy as np
import librosa

# Initialize the model pipeline
pipe = transformers.pipeline(model="fixie-ai/ultravox-v0_3", trust_remote_code=True)

# Load the audio file
path = "transcribing_1.mp3"  # Path to your audio file
audio, sr = librosa.load(path, sr=16000)

# Define maximum allowed length for mel spectrogram (e.g., 3000 samples)
max_length = 3000
hop_length = int(sr * max_length / sr)  # Calculate hop size for chunking

# Split audio into chunks
chunk_size = hop_length * max_length
num_chunks = int(np.ceil(len(audio) / chunk_size))

# Define conversation turns
turns = [
    {
        "role": "system",
        "content": "You are a friendly and helpful character. You love to answer questions for people."
    },
]

# Process each chunk and collect results
results = []
for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size
    chunk = audio[start_idx:end_idx]
    
    # Make sure the chunk length doesn't exceed max_length
    if len(chunk) > max_length:
        chunk = chunk[:max_length]
    
    # Process the chunk through the pipeline
    result = pipe({'audio': chunk, 'turns': turns, 'sampling_rate': sr}, max_new_tokens=30)
    results.append(result)

# Combine the results
final_output = " ".join([res["generated_text"] for res in results])

# Print the final output
print(final_output)
