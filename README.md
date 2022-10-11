# Serving Whisper with BentoML

[Whisper](https://openai.com/blog/whisper/) is a general-purpose speech recognition model. that can perform multilingual speech recognition as well as speech translation and language identification.


## Prerequisites
whisper는 python 3.9.9 그리고 Pytorch 1.10.1에서 학습 및 테스트 하였으며, 코드는 Python 3.7 이상 및 최근 Pytorch 버전과 호환 될것으로 기대됩니다.
```bash
# 1. build venture environment
python3 -m venv venv && . venv/bin/activate
# 2. install whisper
pip install git+https://github.com/openai/whisper.git 
# 3. install requirements
pip install -r requirements.txt
```

## Available models and languages

There are five model sizes, four with English-only versions, offering speed and accuracy tradeoffs. Below are the names of the available models and their approximate memory requirements and relative speed. 


|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |
