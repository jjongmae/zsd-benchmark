"""
Grounding DINO 이미지 추론 스크립트
input 폴더의 모든 이미지에서 텍스트 프롬프트를 사용하여 객체를 감지하고 output 폴더에 결과를 저장합니다.
"""

import os
import glob
import cv2
import torch
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Hugging Face 캐시 디렉토리를 프로젝트 폴더로 설정
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR


# ==================== 설정 상수 ====================
# .env 파일에서 설정 값 로드 (기본값 설정)
TEXT_PROMPT = os.getenv("TEXT_PROMPT", "traffic cone . box .")
BOX_THRESHOLD = float(os.getenv("BOX_THRESHOLD", "0.35"))
TEXT_THRESHOLD = float(os.getenv("TEXT_THRESHOLD", "0.25"))

# 입력 이미지 폴더 경로
INPUT_DIR = "input"

# 출력 결과 폴더 경로
OUTPUT_DIR = "output_gdino"

# 모델 설정 (Hugging Face에서 자동 다운로드)
MODEL_ID = "IDEA-Research/grounding-dino-base"  # 또는 "IDEA-Research/grounding-dino-base"
# 참고: 모델은 위에 설정된 CACHE_DIR (models/ 폴더)에 다운로드됩니다

# 지원하는 이미지 확장자
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
# ================================================


def setup_model(model_id=MODEL_ID, device=None):
    """
    Grounding DINO 모델을 로드합니다 (Hugging Face에서 자동 다운로드).

    Args:
        model_id: Hugging Face 모델 ID
        device: 사용할 디바이스 (cuda/cpu)

    Returns:
        model: 로드된 Grounding DINO 모델
        processor: 이미지 전처리 프로세서
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 캐시 디렉토리 생성
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"모델 로드 중... (처음 실행 시 다운로드에 시간이 걸릴 수 있습니다)")
    print(f"모델 ID: {model_id}")
    print(f"모델 저장 위치: {CACHE_DIR}")

    start_time = time.time()
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=CACHE_DIR)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, cache_dir=CACHE_DIR).to(device)
    load_time = time.time() - start_time

    print(f"모델 로드 완료! (디바이스: {device})")
    print(f"모델 로드 시간: {load_time:.2f}초")

    return model, processor


def run_inference(model, processor, image_path, text_prompt, box_threshold=0.35, text_threshold=0.25, output_dir="output_gdino"):
    """
    이미지에 대해 Grounding DINO 추론을 실행하고 결과를 저장합니다.

    Args:
        model: Grounding DINO 모델
        processor: 이미지 전처리 프로세서
        image_path: 입력 이미지 경로
        text_prompt: 감지할 객체를 설명하는 텍스트 프롬프트 (예: "cat . dog . person")
        box_threshold: 바운딩 박스 신뢰도 임계값
        text_threshold: 텍스트 매칭 신뢰도 임계값
        output_dir: 결과 이미지 저장 디렉토리

    Returns:
        results: 감지 결과
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 전체 추론 시간 측정 시작
    total_start_time = time.time()

    # 이미지 로드
    print(f"이미지 로드 중: {os.path.basename(image_path)}")
    image = Image.open(image_path).convert("RGB")

    # 추론 실행
    print(f"추론 실행 중...")
    print(f"텍스트 프롬프트: '{text_prompt}'")
    print(f"Box threshold: {box_threshold}, Text threshold: {text_threshold}")

    # 입력 전처리
    preprocess_start = time.time()
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(model.device)
    preprocess_time = time.time() - preprocess_start

    # 추론
    inference_start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.time() - inference_start

    # 후처리 (threshold는 결과 필터링 시 적용)
    postprocess_start = time.time()
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[image.size[::-1]]
    )[0]

    # threshold 적용하여 결과 필터링
    keep_indices = results['scores'] >= box_threshold
    results = {
        'scores': results['scores'][keep_indices],
        'labels': [label for i, label in enumerate(results['labels']) if keep_indices[i]],
        'boxes': results['boxes'][keep_indices]
    }
    postprocess_time = time.time() - postprocess_start

    # 전체 추론 시간 계산
    total_inference_time = time.time() - total_start_time

    # 결과 출력
    print(f"감지된 객체 수: {len(results['scores'])}")
    for i, (score, label, box) in enumerate(zip(results['scores'], results['labels'], results['boxes'])):
        print(f"  [{i+1}] {label}: {score:.2f}")

    # 시간 측정 결과 출력
    print(f"\n[시간 측정]")
    print(f"  전처리 시간: {preprocess_time:.4f}초")
    print(f"  추론 시간: {inference_time:.4f}초")
    print(f"  후처리 시간: {postprocess_time:.4f}초")
    print(f"  전체 시간: {total_inference_time:.4f}초")

    # 결과 이미지에 어노테이션 추가
    annotated_image = draw_annotations(image, results)

    # 결과 저장
    output_filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"result_{output_filename}")
    annotated_image.save(output_path)
    print(f"결과 저장 완료: {output_path}")

    return results


def draw_annotations(image, results):
    """
    이미지에 바운딩 박스와 레이블을 그립니다.

    Args:
        image: PIL Image 객체
        results: 모델 추론 결과

    Returns:
        annotated_image: 어노테이션이 추가된 이미지
    """
    draw = ImageDraw.Draw(image)

    # 폰트 설정 (기본 폰트 사용)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    # 분홍색으로 고정
    color = "pink"

    for idx, (score, label, box) in enumerate(zip(results['scores'], results['labels'], results['boxes'])):
        box = [round(i, 2) for i in box.tolist()]
        x1, y1, x2, y2 = box

        # 바운딩 박스 그리기
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 레이블 텍스트 (score 없이 라벨만)
        text = label

        # 텍스트 배경
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(text_bbox, fill=color)

        # 텍스트 그리기
        draw.text((x1, y1), text, fill="white", font=font)

    return image


def get_image_files(input_dir):
    """
    입력 디렉토리에서 지원하는 모든 이미지 파일을 찾습니다.

    Args:
        input_dir: 이미지를 찾을 디렉토리

    Returns:
        image_files: 이미지 파일 경로 리스트
    """
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        # 대소문자 모두 검색
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))

    # 중복 제거 (Windows는 대소문자 구분 안함)
    image_files = list(set(image_files))

    return sorted(image_files)


def main():
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"Grounding DINO 이미지 추론 시작")
    print("=" * 60)
    print(f"사용 디바이스: {device}")
    print(f"텍스트 프롬프트: '{TEXT_PROMPT}'")
    print(f"입력 폴더: {INPUT_DIR}")
    print(f"출력 폴더: {OUTPUT_DIR}")
    print(f"Box threshold: {BOX_THRESHOLD}, Text threshold: {TEXT_THRESHOLD}")
    print("=" * 60)

    # 입력 디렉토리 확인
    if not os.path.exists(INPUT_DIR):
        print(f"\n오류: 입력 폴더 '{INPUT_DIR}'가 존재하지 않습니다.")
        print(f"'{INPUT_DIR}' 폴더를 생성하고 이미지를 넣어주세요.")
        return

    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 이미지 파일 찾기
    image_files = get_image_files(INPUT_DIR)

    if not image_files:
        print(f"\n오류: '{INPUT_DIR}' 폴더에 이미지 파일이 없습니다.")
        print(f"지원하는 형식: {', '.join(SUPPORTED_EXTENSIONS)}")
        return

    print(f"\n찾은 이미지 수: {len(image_files)}")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")

    # 모델 로드
    print("\n" + "=" * 60)
    model, processor = setup_model(MODEL_ID, device)
    print("=" * 60)

    # 모든 이미지에 대해 추론 실행
    total_images = len(image_files)
    all_start_time = time.time()

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{total_images}] 처리 중: {os.path.basename(image_path)}")
        print("-" * 60)

        try:
            run_inference(
                model=model,
                processor=processor,
                image_path=image_path,
                text_prompt=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                output_dir=OUTPUT_DIR
            )
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - all_start_time
    avg_time = total_time / total_images if total_images > 0 else 0

    print("\n" + "=" * 60)
    print(f"모든 이미지 추론 완료!")
    print(f"총 처리된 이미지: {total_images}")
    print(f"전체 처리 시간: {total_time:.2f}초")
    print(f"평균 처리 시간: {avg_time:.2f}초/이미지")
    print(f"결과 저장 위치: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
