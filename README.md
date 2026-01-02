# zsd-benchmark

Grounding DINO를 사용한 제로샷 객체 탐지(Zero-Shot Object Detection) 벤치마크 프로젝트

## 설치 방법

### 1. 가상환경 활성화

```bash
# Windows Git Bash
source venv/Scripts/activate

# Windows CMD
venv\Scripts\activate

# Windows PowerShell
venv\Scripts\Activate.ps1
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

**참고:** 모델은 처음 실행 시 Hugging Face에서 자동으로 다운로드되어 프로젝트의 `models/` 폴더에 저장됩니다. 별도의 가중치 파일 다운로드는 필요하지 않습니다!

## 사용 방법

### 1. 이미지 준비

`input` 폴더를 생성하고 추론할 이미지들을 넣어주세요:

```bash
mkdir input
# input 폴더에 이미지 파일들을 복사
```

### 2. 텍스트 프롬프트 설정 (선택사항)

`grounding_dino_inference.py` 파일 상단의 `TEXT_PROMPT` 상수를 원하는 객체로 수정할 수 있습니다:

```python
# 기본값
TEXT_PROMPT = "person . car . bicycle . motorcycle . bus . truck . traffic light . stop sign"

# 예시: 동물 감지
TEXT_PROMPT = "cat . dog . bird . horse"
```

### 3. 스크립트 실행

```bash
python grounding_dino_inference.py
```

스크립트는 `input` 폴더의 모든 이미지를 처리하고 결과를 `output` 폴더에 저장합니다.

### 설정 변경

스크립트 상단의 상수들을 수정하여 동작을 변경할 수 있습니다:

- `TEXT_PROMPT`: 감지할 객체 (점(.)으로 구분)
- `INPUT_DIR`: 입력 이미지 폴더 경로 (기본값: "input")
- `OUTPUT_DIR`: 결과 저장 폴더 경로 (기본값: "output")
- `BOX_THRESHOLD`: 바운딩 박스 신뢰도 임계값 (기본값: 0.35)
- `TEXT_THRESHOLD`: 텍스트 매칭 신뢰도 임계값 (기본값: 0.25)
- `SUPPORTED_EXTENSIONS`: 지원하는 이미지 확장자 리스트

## 프로젝트 구조

```
zsd-benchmark/
├── venv/                          # 가상환경
├── models/                        # Hugging Face 모델 캐시 (자동 생성)
│   └── (다운로드된 모델 파일들)
├── input/                         # 입력 이미지 폴더 (사용자가 생성)
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── output/                        # 추론 결과 저장 폴더 (자동 생성)
│   ├── result_image1.jpg
│   ├── result_image2.png
│   └── ...
├── grounding_dino_inference.py   # 메인 추론 스크립트
├── requirements.txt              # 패키지 의존성
├── .gitignore                    # Git 무시 파일
└── README.md                     # 프로젝트 설명
```

**참고:** 모델 파일은 처음 실행 시 Hugging Face에서 자동으로 다운로드되어 프로젝트의 `models/` 폴더에 저장됩니다.

## 예제

### 기본 실행
```bash
# input 폴더의 모든 이미지 처리 (기본 프롬프트 사용)
python grounding_dino_inference.py
```

### 텍스트 프롬프트 변경 예시

스크립트 파일에서 `TEXT_PROMPT`를 수정:

```python
# 동물 감지
TEXT_PROMPT = "cat . dog . bird . horse . cow . sheep"

# 실내 객체 감지
TEXT_PROMPT = "chair . table . sofa . bed . tv . laptop . book"

# 음식 감지
TEXT_PROMPT = "pizza . burger . sandwich . cake . coffee . apple . banana"
```

## 참고

- [Grounding DINO GitHub](https://github.com/IDEA-Research/GroundingDINO)
- [Grounding DINO Paper](https://arxiv.org/abs/2303.05499)