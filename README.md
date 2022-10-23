# Serving Whisper with BentoML

[Whisper](https://openai.com/blog/whisper/) is a general-purpose speech recognition model. that can perform multilingual speech recognition as well as speech translation and language identification.

[Whisper](https://openai.com/blog/whisper/) 는 범용적 speeck recognition model이며 다양한 언어에서 효과를 내며 이해하고 번역도 잘합니다.
<!-- 진행 내용 설명  -->
<!-- 이러한 whisper를 많은 사람들이 이용할 수 있도록 웹서비스로 만들기 위해 진행되는 내용들 입니다. BentoML은 이 과정 중 ML service를 손쉽게 API로 만들 수 있어 이를 활용하여 배포하도록 하겠습니다. -->

## Prerequisites
whisper는 python 3.9.9 그리고 Pytorch 1.10.1에서 학습 및 테스트 하였으며, 코드는 Python 3.7 이상 및 최근 Pytorch 버전과 호환 될 것 입니다.
```bash
# 1. build venture environment
python3 -m venv venv && . venv/bin/activate
# 2. install requirements
pip install -r requirements.txt
```
<!-- BentoML 사용 내용 -->
<!-- service.py에 모델 정의 부분 with whisper 모듈을 사용법 -->
<!-- audio 파일을 받아오는 부분 -->

<!-- 이를 실행하는 방법 -->
<!-- configuration.yaml 내용과 목적 -->
<!-- bentoml servce service:svc --production -->
<!-- bentoml servce service:svc --reload -->
<!-- 환경 config 생성 bentoml.yaml 내용과 목적 -->


## Available models and languages

There are five model sizes, four with English-only versions, offering speed and accuracy tradeoffs. Below are the names of the available models and their approximate memory requirements and relative speed. 

5가지 모델 크기가 있으며, 4가지 모델에는 영어 전용 버전이 포함되어 있어 속도와 정확성이 tradeoff가 있습니다. 아래는 사용 가능한 모델의 이름과 대략적인 메모리 요구 사항 및 상대 속도 입니다.


|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |

## Streamlit use
```bash
cd app
streamlit run main.py
```
![record-video](/img/record_streamlit.gif)
