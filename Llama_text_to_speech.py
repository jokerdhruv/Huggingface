# from langchain_community.llms import LlamaCpp
# from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
# from langchain_core.prompts import PromptTemplate


# template = """Question: {question}

# Answer: Let's work this out in a step by step way to be sure we have the right answer."""

# prompt = PromptTemplate.from_template(template)


# # Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


# # Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path="llama-2-7b.Q2_K.gguf",
#     temperature=0.75,
#     max_tokens=2000,
#     top_p=1,
#     callback_manager=callback_manager,
#     verbose=True,  # Verbose is required to pass to the callback manager
# )

# question = """
# Can u convert audio file to text if i provide you the file ?
# """
# llm.invoke(question)



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

str = ''.join(transcription_with_special_tokens)
print ('THIS IS THE EXTRACTED STRING:----- ', str)
print("Transcription with special tokens:", transcription_with_special_tokens)
print("Transcription without special tokens:", transcription_without_special_tokens)



# Step 1: Use Whisper to transcribe audio to text (this part is already done in your code)
# Assuming transcription_without_special_tokens contains the transcribed text from Whisper
transcribed_text = ''.join(transcription_without_special_tokens)  # Combine the text if it's in a list

print("Transcribed Text: ", transcribed_text)  # Print the transcribed text

# Step 2: Use Llama to generate word embeddings from the transcribed text

# Load the Llama model for word embeddings (replace with your Llama model path)
llama = Llama(
    model_path="path/to/your/llama-model.bin",  # Replace with the actual model path to your Llama model
    n_ctx=2048,
    n_gpu_layers=0,  # Set this based on whether you're using GPU or CPU
    embedding=True  # Enable word embedding functionality
)

# Generate word embeddings from the transcribed text
embedding = llama.embed(transcribed_text)

# Print or process the generated embedding
print("Word Embedding: ", embedding)
