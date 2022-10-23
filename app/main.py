import streamlit as st
import io
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

import whisper

import torch

# pre-process
# file object input case
def trans_byte2arr(byte_data: bytes):

    arr_data, _ = sf.read(file=io.BytesIO(byte_data.read()), dtype="float32")

    sig_data = merge_sig(arr_data)

    return sig_data


def merge_sig(arr_data):

    if arr_data.ndim == 2:
        # left right channel sound file case
        # element-wise add left and right
        sig_data = arr_data.sum(axis=1)
    elif arr_data.ndim > 2:
        print("this file is not audio file")
    else:
        return arr_data

    return sig_data


# pre-process
def audio_speed_reduce(sig_data: np.array, sample_rate: int):
    if sample_rate > 16000:
        reduce_size = sample_rate / 16000
    elif sample_rate < 16000:
        reduce_size = 16000 / sample_rate
    else:
        reduce_size = None

    sig_data = merge_sig(sig_data)

    if reduce_size is None:
        return audio
    else:
        try:
            audio = sig_data.reshape(-1, int(reduce_size)).mean(axis=1)
        except:
            slice_size = len(sig_data) % reduce_size
            audio = (
                sig_data[: -int(slice_size)].reshape(-1, int(reduce_size)).mean(axis=1)
            )

    return audio


def convert_byte_audio(byte_data):
    # convert audio from bytes
    arr_data, sr = sf.read(file=io.BytesIO(byte_data), dtype="float32")

    # reduce audio
    audio = audio_speed_reduce(arr_data, sr)
    return audio


def get_langage_cls(audio_arr: np.array, model: torch.nn.Module):

    # data slice 30 sec
    audio = whisper.pad_or_trim(audio_arr)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # detect the spoken language
    _, probs = model.detect_language(mel)

    return probs


def transcribe(audio: np.array, model: torch.nn.Module, task: str = "transcribe"):

    base_option = dict(beam_size=5, best_of=5)

    if task == "transcribe":
        base_option = dict(task="transcribe", **base_option)
    else:
        base_option = dict(task="translate", **base_option)

    result = model.transcribe(audio, **base_option)
    return result["text"]


def load_model(model_name: str):
    model = whisper.load_model(model_name)
    return model


file_data = st.file_uploader("Upload your audio file")


if file_data is not None:
    # To read file as bytes:
    bytes_data = file_data.getvalue()

    audio_arr = convert_byte_audio(bytes_data)

    # audio plotting
    fig, ax = plt.subplots()
    ax.plot(audio_arr)
    st.pyplot(fig)

    st.audio(bytes_data)

    model_option = [
        "tiny",
        "base",
        "small",
        "medium",
        "large",
    ]
    selected_model_size = st.selectbox(
        "What do you want model size?", ["None"] + model_option
    )

    if selected_model_size in model_option:
        model = load_model(selected_model_size)

        lang_button = st.button("What is langage")
        if lang_button:
            probs = get_langage_cls(audio_arr=audio_arr, model=model)
            st.write(f"Detected language: {max(probs, key=probs.get)}")

        task_option = ["transcribe", "translate"]
        translate_task = st.selectbox("What is your task", ["None"] + task_option)

        if translate_task != "None":
            result = transcribe(audio=audio_arr, model=model, task=translate_task)
            st.write(result)
