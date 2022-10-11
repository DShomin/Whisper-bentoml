import whisper

# model = whisper.load_model("medium")

model = whisper.load_model("base")
audio = whisper.load_audio("../audio_sample.wav")
print(f"1. {audio.shape}")
print("audio sample")
print(audio[166000:166100])
# 15115200  5038400
options = dict(beam_size=5, best_of=5)
# transcribe_options = dict(task="transcribe", **options)
translate_options = dict(task="translate", **options)

result = model.transcribe(audio, **translate_options)
print(result["text"])

# print(result["text"])

# load audio and pad/trim it to fit 30 seconds
# audio = whisper.load_audio("../audio_sample.m4a")
# audio = whisper.pad_or_trim(audio)
# print(audio.shape)

# # make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio).to(model.device)
# print(mel.shape)

# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# # decode the audio
# options = whisper.DecodingOptions(task="translate")
# result = whisper.decode(model, mel, options)

# # print the recognized text
# print(result.text)
