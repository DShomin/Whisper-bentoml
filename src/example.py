import whisper


# model = whisper.load_model("medium")
model = whisper.load_model("base")
audio = whisper.load_audio("../audio_sample.wav")
options = dict(beam_size=5, best_of=5)
# transcribe_options = dict(task="transcribe", **options)
translate_options = dict(task="translate", **options)

result = model.transcribe(audio, **translate_options)
print(result["text"])