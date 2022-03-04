# 🏆OCR_ancient_korean
2021 옛한글 OCR 인공지능 학습데이터 해커톤

# Task Description

### Subject
본 대회의 주제는 옛한글로 쓰인 고문헌 이미지를 데이터화하고 이를 바탕으로 효율적인 서비스를 제공하는 것이었습니다. 옛한글로 이루어진 이미지는 일반 한글과 다르게 많은 자모음이 추가되기 때문에 단순 분류 문제가 아닌 인식 문제로 해결해야 합니다. 또한 일반적인 언어와 다르게 합성어기 때문에 각 단어를 구분하는 것이 아닌 자음 모음 받침의 조합으로 분석하는 것이 적절하다고 판단하였습니다.

이를 구현하기 위해 글자를 추출하는 Object Detection 모델을 활용하였고 추출된 문자를 자음 모음 받침의 조합으로 인식하는 Recognition 모델로 구성하였습니다.

### Data
- 학습 데이터 : 목판본, 필사본, 활자본으로 이루어진 54000장의 데이터
- 검증 데이터 : 목판본, 필사본, 활자본으로 이루어진 6000장의 데이터

### Metric

- 평가 척도
  - Object Detection : IoU
  - Recognition : Word accuracy ( 자음, 모음, 받침이 모두 맞을 때만 맞은 것으로 평가 )

# Project Result
- 대상 400만원

### Coding Explanation
- Detection : MMDetection 패키지에 맞추어 작성된 코드입니다.
- Recognition : Google Colab의 ssh 환경에 맞추어 작성된 코드입니다.

### Data Structure
- Detection
```python
[code]
├── mmdetection/ # mmdetection package
├── mmdet.ipynb 
└── requirements.txt
```

- Recognition
```python
[code]
├── scheduler.py
├── model.py # SWIN TRN
├── utils.py # useful utilities
├── dataset.py # modules related to data
├── requirements.txt
├── main.py # train code
└── inference.py
```
