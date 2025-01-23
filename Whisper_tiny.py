# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from datasets import load_dataset

# # load model and processor
# processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
# model.config.forced_decoder_ids = None

# # load dummy dataset and read audio files
# ds = load_dataset("audio", "clean", split="validation")
# sample = ds[0]["audio"]
# input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

# # generate token ids
# predicted_ids = model.generate(input_features)
# # decode token ids to text
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
# model.config.forced_decoder_ids = WhisperProcessor.get_decoder_prompt_ids(language="english", task="transcribe")
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


