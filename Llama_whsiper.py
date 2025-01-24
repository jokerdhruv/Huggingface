# from llama_cpp import Llama
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# import torchaudio

# # load model and processor
# processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
# model.config.forced_decoder_ids = None

# # Load your custom audio file
# audio_path = "audio.mp3"  # Replace with the path to your audio file
# waveform, original_sample_rate = torchaudio.load(audio_path)

# # Resample the audio to 16,000 Hz if necessary
# target_sample_rate = 16000
# if original_sample_rate != target_sample_rate:
#     resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
#     waveform = resampler(waveform)

# # Process the audio file
# input_features = processor(waveform.squeeze().numpy(), sampling_rate=target_sample_rate, return_tensors="pt").input_features

# # Generate token ids
# predicted_ids = model.generate(input_features)

# # Decode token ids to text
# transcription_with_special_tokens = processor.batch_decode(predicted_ids, skip_special_tokens=False)
# transcription_without_special_tokens = processor.batch_decode(predicted_ids, skip_special_tokens=True)

# # Combine the transcription list into a single string
# transcribed_text = ''.join(transcription_without_special_tokens)

# print("Transcribed Text: ", transcribed_text)  # Print the transcribed text

# # Step 2: Use Llama to generate word embeddings from the transcribed text

# # Load the Llama model for word embeddings (replace with your Llama model path)
# llama = Llama(
#     model_path="llama-2-7b.Q2_K.gguf",  # Replace with the actual model path to your Llama model
#     n_ctx=2048,
#     n_gpu_layers=0,  # Set this based on whether you're using GPU or CPU
#     embedding=True  # Enable word embedding functionality
# )

# # Generate word embeddings from the transcribed text
# embedding = llama.embed(transcribed_text)

# # Print or process the generated embedding
# print("Word Embedding: ", embedding)






# from llama_cpp import Llama
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# import torchaudio
# import numpy as np

# # load model and processor
# processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
# model.config.forced_decoder_ids = None

# # Load your custom audio file
# audio_path = "audio.mp3"  # Replace with the path to your audio file
# waveform, original_sample_rate = torchaudio.load(audio_path)

# # Resample the audio to 16,000 Hz if necessary
# target_sample_rate = 16000
# if original_sample_rate != target_sample_rate:
#     resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
#     waveform = resampler(waveform)

# # Process the audio file
# input_features = processor(waveform.squeeze().numpy(), sampling_rate=target_sample_rate, return_tensors="pt").input_features

# # Generate token ids
# predicted_ids = model.generate(input_features)

# # Decode token ids to text
# transcription_with_special_tokens = processor.batch_decode(predicted_ids, skip_special_tokens=False)
# transcription_without_special_tokens = processor.batch_decode(predicted_ids, skip_special_tokens=True)

# # Combine the transcription list into a single string
# transcribed_text = ''.join(transcription_without_special_tokens)

# print("Transcribed Text: ", transcribed_text)  # Print the transcribed text

# # Step 2: Use Llama to generate word embeddings from the transcribed text


# llama = Llama(
#     model_path="llama-2-7b.Q2_K.gguf",  # Replace with the actual model path to your Llama model
#     n_ctx=2048,
#     n_gpu_layers=0,  # Set this based on whether you're using GPU or CPU
#     embedding=True  # Enable word embedding functionality
# )

# # Split the text into words (or tokens) for better representation
# words = transcribed_text.split()  # You can adjust this based on how you want to split (e.g., by punctuation, spaces, etc.)

# # Generate word embeddings for each word
# word_embeddings = []
# for word in words:
#     embedding = llama.embed(word)  # Get the embedding for each word
#     word_embeddings.append(embedding)

# # Present the output in a structured way
# for word, embedding in zip(words, word_embeddings):
#     print(f"Word: '{word}'")
#     print(f"Embedding (Vector): {embedding}")
#     print("\n---\n")

from llama_cpp import Llama
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = None

# Load your custom audio file
audio_path = "audio.mp3"  # Replace with the path to your audio file
waveform, original_sample_rate = torchaudio.load(audio_path)

# Resample the audio to 16,000 Hz if necessary
target_sample_rate = 16000
if original_sample_rate != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
    waveform = resampler(waveform)

# Process the audio file
input_features = processor(waveform.squeeze().numpy(), sampling_rate=target_sample_rate, return_tensors="pt").input_features

# Generate token ids
predicted_ids = model.generate(input_features)

# Decode token ids to text
transcription_with_special_tokens = processor.batch_decode(predicted_ids, skip_special_tokens=False)
transcription_without_special_tokens = processor.batch_decode(predicted_ids, skip_special_tokens=True)

# Combine the transcription list into a single string
transcribed_text = ''.join(transcription_without_special_tokens)

print("Transcribed Text: ", transcribed_text)  # Print the transcribed text

# Step 2: Use Llama to generate word embeddings from the transcribed text
try:
    llama = Llama(
        model_path="llama-2-7b.Q2_K.gguf",  # Replace with the actual model path to your Llama model
        n_ctx=2048  ,
        n_gpu_layers=0,  # Set this based on whether you're using GPU or CPU
        embedding=True  # Enable word embedding functionality
    )

    # Split the text into words (or tokens) for better representation
    words = transcribed_text.split()  # You can adjust this based on how you want to split (e.g., by punctuation, spaces, etc.)

    # Generate word embeddings for each word
    word_embeddings = []
    for word in words:
        embedding = llama.embed(word)  # Get the embedding for each word
        word_embeddings.append(embedding)

    # Present the output in a structured way
    for word, embedding in zip(words, word_embeddings):
        print(f"Word: '{word}'")
        print(f"Embedding (Vector): {embedding}")
        print("\n---\n")

finally:
    # Ensure proper cleanup of the llama model
    if 'llama' in locals() and llama is not None:
        try:
            del llama
            print("Llama model cleanup complete.")
        except Exception as e:
            print(f"Error during model cleanup: {e}")
