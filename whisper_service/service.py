import bentoml

from bentoml.io import File, Text
import soundfile as sf
import io

import whisper

import torch


class WhisperRunneabel(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self) -> None:
        MODEL_TYPE = "medium"  # base
        self.model = whisper.load_model(MODEL_TYPE)
        base_option = dict(beam_size=5, best_of=5)

        self.transcribe_options = dict(task="transcribe", **base_option)
        self.translate_options = dict(task="translate", **base_option)

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def transcribe(self, audio):
        result = self.model.transcribe(audio, **self.transcribe_options)
        return result

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def translate(self, audio):
        result = self.model.transcribe(audio, **self.translate_options)
        return result


whisper_runner = bentoml.Runner(
    WhisperRunneabel, name="whisper_runner", max_batch_size=10
)

svc = bentoml.Service("whisper", runners=[whisper_runner])


def convert_byte_audio(data):
    # convert audio from bytes
    arr_data, sr = sf.read(file=io.BytesIO(data.read()), dtype="float32")
    single_audio = arr_data[..., 0].copy()
    audio = torch.from_numpy(single_audio).float()

    # reduce audio
    audio = audio.reshape(-1, 3).mean(dim=1)
    return audio


# Full size


@svc.api(input=File(), output=Text())
def transcribe(data):

    audio = convert_byte_audio(data)

    result = whisper_runner.transcribe.run(audio)
    return result["text"]


@svc.api(input=File(), output=Text())
def translate(data):

    audio = convert_byte_audio(data)

    result = whisper_runner.translate.run(audio)
    return result["text"]


# sample size 30 sec
@svc.api(input=File(), output=Text())
def transcribe_sample(data):

    audio = convert_byte_audio(data)

    # sub sample audio 30sec
    audio = whisper.audio.pad_or_trim(audio)

    result = whisper_runner.transcribe.run(audio)
    return result["text"]


@svc.api(input=File(), output=Text())
def translate_sample(data):

    audio = convert_byte_audio(data)

    # sub sample audio 30sec
    audio = whisper.audio.pad_or_trim(audio)

    result = whisper_runner.translate.run(audio)
    return result["text"]
