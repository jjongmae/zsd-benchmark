# zsd-benchmark

제로샷 객체 탐지(Zero-Shot Object Detection) 벤치마크 프로젝트

이 프로젝트는 여러 제로샷 객체 탐지 모델을 제공합니다:
- **Grounding DINO**: 이미지에서 객체 탐지
- **LLMDet**: 이미지에서 객체 탐지 (텍스트 프롬프트 기반)
- **LLMDet + BoTSORT**: 비디오에서 객체 탐지 및 추적

## 목차
- [설치 방법](#설치-방법)
- [프로젝트 구조](#프로젝트-구조)
- [Grounding DINO 이미지 추론](#grounding-dino-이미지-추론)
- [LLMDet 이미지 추론](#llmdet-이미지-추론)
- [LLMDet 비디오 트래킹](#llmdet-비디오-트래킹-llmdet--botsort)
- [성능 최적화](#성능-최적화)
- [트러블슈팅](#트러블슈팅)
- [참고](#참고)

---

## 설치 방법

### 1. 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv .venv

# Linux/Mac
source .venv/bin/activate

# Windows Git Bash
source .venv/Scripts/activate

# Windows CMD
.venv\Scripts\activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

**참고:** 모델은 처음 실행 시 Hugging Face에서 자동으로 다운로드되어 프로젝트의 `models/` 폴더에 저장됩니다. 별도의 가중치 파일 다운로드는 필요하지 않습니다!

### 3. 환경 변수 설정 (선택사항)

```bash
# .env_example을 .env로 복사
cp .env_example .env

# .env 파일을 편집하여 설정 변경
```

---

## 프로젝트 구조

```
zsd-benchmark/
├── .venv/                         # Python 가상환경
├── models/                        # Hugging Face 모델 캐시 (자동 생성)
│   └── (다운로드된 모델 파일들)
├── input/                         # 입력 이미지 폴더 (사용자가 생성)
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── output_gdino/                  # Grounding DINO 결과 저장 폴더 (자동 생성)
│   └── result_*.jpg
├── output_llmdet/                 # LLMDet 결과 저장 폴더 (자동 생성)
│   └── result_*.jpg
├── output/                        # 비디오 트래킹 결과 저장 폴더 (자동 생성)
│   └── *_analyzed.mp4
├── gdino.py                       # Grounding DINO 이미지 추론 스크립트
├── llmdet.py                      # LLMDet 이미지 추론 스크립트
├── llmdet-video-tracking.py      # LLMDet + BoTSORT 비디오 트래킹 스크립트
├── requirements.txt              # 패키지 의존성
├── .env                          # 환경 변수 설정 (사용자가 생성)
├── .env_example                  # 환경 변수 설정 예시
├── .gitignore                    # Git 무시 파일
└── README.md                     # 프로젝트 설명 (이 파일)
```

---

## Grounding DINO 이미지 추론

### 소개
`gdino.py`는 Grounding DINO 모델을 사용하여 이미지에서 객체를 감지하는 스크립트입니다.

### 사용 방법

1. **이미지 준비**: `input` 폴더를 생성하고 이미지를 넣어주세요

```bash
mkdir input
# input 폴더에 이미지 파일들을 복사
```

2. **텍스트 프롬프트 설정**: `.env` 파일이나 `gdino.py` 파일에서 `TEXT_PROMPT` 수정

3. **스크립트 실행**:

```bash
python gdino.py
```

### 설정

#### 방법 1: .env 파일 사용 (권장)

```env
TEXT_PROMPT=person . car . bicycle . motorcycle . bus . truck .
BOX_THRESHOLD=0.35
TEXT_THRESHOLD=0.25
```

#### 방법 2: 소스 코드 수정

`gdino.py` 파일의 상수를 직접 수정:

```python
TEXT_PROMPT = "cat . dog . bird . horse"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
INPUT_DIR = "input"
OUTPUT_DIR = "output_gdino"
```

### 출력

결과는 `output_gdino` 폴더에 `result_` 접두사가 붙은 파일명으로 저장됩니다.

**예시:**
- 입력: `input/photo.jpg`
- 출력: `output_gdino/result_photo.jpg`

---

## LLMDet 이미지 추론

### 소개
`llmdet.py`는 LLMDet(Large Language Model based Object Detection) 모델을 사용하여 이미지에서 객체를 감지하는 스크립트입니다.

### 사용 방법

1. **이미지 준비**: `input` 폴더에 이미지를 넣어주세요
2. **텍스트 프롬프트 설정**: `.env` 파일이나 소스코드에서 `TEXT_PROMPT` 수정
3. **스크립트 실행**:

```bash
python llmdet.py
```

### 설정 (.env 파일)

`.env` 파일을 생성하여 설정을 변경할 수 있습니다:

```env
TEXT_PROMPT=traffic cone . box .
BOX_THRESHOLD=0.35
TEXT_THRESHOLD=0.25
```

### 출력

결과는 `output_llmdet` 폴더에 `result_` 접두사가 붙은 파일명으로 저장됩니다.

**예시:**
- 입력: `input/photo.jpg`
- 출력: `output_llmdet/result_photo.jpg`

### 예제

```bash
# 기본 실행
python llmdet.py

# 결과는 output_llmdet 폴더에 저장됩니다
```

---

## LLMDet 비디오 트래킹 (LLMDet + BoTSORT)

### 소개
`llmdet-video-tracking.py`는 LLMDet과 BoTSORT 트래커를 결합하여 비디오에서 객체를 감지하고 추적하는 스크립트입니다.

### 주요 기능
- ✨ **LLMDet 추론**: 자연어 텍스트 프롬프트를 사용하여 객체 감지
- 🎯 **BoTSORT 트래킹**: boxmot 라이브러리를 사용한 고성능 객체 추적
- 🎬 **비디오 처리**: 비디오 파일 입력 및 분석 결과 비디오 출력
- 📊 **실시간 진행 상황**: 처리 Progress 및 FPS 표시

### 사용 방법

#### 기본 사용
```bash
python llmdet-video-tracking.py input_video.mp4
```

#### 커스텀 프롬프트 사용
```bash
python llmdet-video-tracking.py input_video.mp4 --prompt "person . car . bicycle ."
```

#### 임계값 조정
```bash
python llmdet-video-tracking.py input_video.mp4 --box-threshold 0.4 --text-threshold 0.3
```

#### 출력 디렉토리 지정
```bash
python llmdet-video-tracking.py input_video.mp4 --output-dir my_output
```

### 명령행 인자

| 인자               | 설명                                   | 기본값                 |
| ------------------ | -------------------------------------- | ---------------------- |
| `video`            | 입력 비디오 파일 경로 (필수)           | -                      |
| `--prompt`         | 감지할 객체를 설명하는 텍스트 프롬프트 | "traffic cone . box ." |
| `--box-threshold`  | 바운딩 박스 신뢰도 임계값              | 0.35                   |
| `--text-threshold` | 텍스트 매칭 신뢰도 임계값              | 0.25                   |
| `--output-dir`     | 출력 비디오 저장 디렉토리              | output                 |

### 출력

분석된 비디오는 `output` 폴더(또는 지정한 출력 디렉토리)에 2가지 버전으로 저장됩니다:

1. **트래킹 영상 (`_tracking.mp4`)**:
   - BoTSORT 트래킹 ID가 포함된 결과
   - 객체 ID 및 신뢰도 점수 표시 (예: "ID:5 0.87")
   - 각 객체 ID별 고유 색상 바운딩 박스

2. **검지 영상 (`_detection.mp4`)**:
   - 순수 ZSD(Zero-Shot Detection) 결과
   - 트래킹 ID 없이 클래스 이름과 점수만 표시 (예: "person 0.87")
   - 클래스별 고유 색상 바운딩 박스

**예시:**
- 입력: `traffic_video.mp4`
- 출력 1: `output/traffic_video_tracking.mp4`
- 출력 2: `output/traffic_video_detection.mp4`

### 도움말 보기
```bash
python llmdet-video-tracking.py --help
```

---

## 텍스트 프롬프트 작성 가이드

### 기본 형식
객체들을 점(.)으로 구분하여 나열합니다:

```
"객체1 . 객체2 . 객체3 ."
```

### 예제

**1. 교통 관련 객체**
```env
TEXT_PROMPT=person . car . bicycle . motorcycle . bus . truck . traffic light . stop sign .
```

**2. 동물**
```env
TEXT_PROMPT=cat . dog . bird . horse . cow . sheep . elephant . giraffe .
```

**3. 실내 객체**
```env
TEXT_PROMPT=chair . table . sofa . bed . tv . laptop . book . clock .
```

**4. 음식**
```env
TEXT_PROMPT=pizza . burger . sandwich . cake . coffee . apple . banana . orange .
```

**5. 안전 장비**
```env
TEXT_PROMPT=traffic cone . safety vest . helmet . barrier . warning sign .
```

---

## 성능 최적화

### GPU 사용 확인
CUDA가 사용 가능한 경우 자동으로 GPU를 사용합니다:

```bash
# GPU 사용 확인
python -c "import torch; print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"
```

### 처리 속도 개선 팁

#### 1. 더 작은 모델 사용
스크립트 파일에서 `MODEL_ID` 변경:
- **Grounding DINO**: `IDEA-Research/grounding-dino-tiny`
- **LLMDet**: `iSEE-Laboratory/llmdet_base` (기본값은 `llmdet_large`)

#### 2. 임계값 조정
- `--box-threshold`를 높여 감지 객체 수 감소 (예: 0.5)
- 불필요한 False Positive 감소

#### 3. 비디오 해상도 감소
ffmpeg를 사용하여 입력 비디오를 사전에 리사이즈:

```bash
ffmpeg -i input_video.mp4 -vf scale=1280:-1 input_video_resized.mp4
```

#### 4. 배치 처리
여러 이미지/비디오를 처리할 때는 모델을 한 번만 로드하므로 효율적입니다.

---

## 트러블슈팅

### 1. boxmot 설치 오류
```bash
pip install boxmot --upgrade
```

### 2. CUDA 메모리 부족 오류
**해결 방법:**
- 더 작은 모델 사용 (`llmdet_base`, `grounding-dino-tiny`)
- 비디오 해상도를 낮춰서 처리
- 배치 크기 감소

### 3. 모델 다운로드 실패
**해결 방법:**
```bash
# Hugging Face 토큰이 필요한 경우
pip install huggingface_hub
huggingface-cli login
```

### 4. ReID 모델 없음 경고 (비디오 트래킹)
**안내:**
ReID 모델 없이도 기본 트래킹이 작동하지만, 더 정확한 추적을 위해서는 ReID 모델 다운로드를 권장합니다.

**ReID 모델 다운로드 (선택사항):**
```bash
# models 폴더 생성
mkdir -p models

# boxmot GitHub releases에서 osnet_x0_25_msmt17.pt 다운로드
# https://github.com/mikel-brostrom/boxmot/releases
```

### 5. 입력 폴더가 없다는 오류
```bash
# input 폴더 생성
mkdir input

# 이미지를 input 폴더에 복사
cp /path/to/your/images/* input/
```

### 6. 지원하지 않는 이미지 형식
**지원하는 이미지 형식:**
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

**지원하는 비디오 형식:**
- `.mp4`, `.avi`, `.mov`, `.mkv` 등 (OpenCV가 지원하는 모든 형식)

---

## 참고 자료

### Grounding DINO
- 📄 [Grounding DINO Paper](https://arxiv.org/abs/2303.05499)
- 💻 [Grounding DINO GitHub](https://github.com/IDEA-Research/GroundingDINO)
- 🤗 [Hugging Face Model](https://huggingface.co/IDEA-Research/grounding-dino-base)

### LLMDet
- 📄 [LLMDet Paper](https://arxiv.org/abs/2407.08033)
- 💻 [LLMDet GitHub](https://github.com/iSEE-Laboratory/LLMDet)
- 🤗 [Hugging Face Models](https://huggingface.co/iSEE-Laboratory)

### BoTSORT
- 📄 [BoT-SORT Paper](https://arxiv.org/abs/2206.14651)
- 💻 [boxmot GitHub](https://github.com/mikel-brostrom/boxmot)
- 📚 [boxmot Documentation](https://github.com/mikel-brostrom/boxmot/wiki)

---

## 라이선스

이 프로젝트는 각 모델의 라이선스를 따릅니다:
- Grounding DINO: Apache 2.0
- LLMDet: Apache 2.0
- boxmot: AGPL-3.0

---

## 기여

버그 리포트, 기능 제안, Pull Request를 환영합니다!