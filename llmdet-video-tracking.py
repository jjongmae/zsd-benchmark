"""
LLMDet 비디오 추론 및 BoTSORT 트래킹 스크립트
비디오 파일에서 텍스트 프롬프트를 사용하여 객체를 감지하고 BoTSORT로 추적하여 결과를 저장합니다.
"""

import os
import sys
import cv2
import torch
import numpy as np
import time
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from dotenv import load_dotenv

# boxmot tracker import
try:
    from boxmot import BotSort
except ImportError:
    print("오류: boxmot 라이브러리를 찾을 수 없습니다.")
    print("다음 명령어로 설치하세요: pip install boxmot")
    sys.exit(1)

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

# 출력 결과 폴더 경로
OUTPUT_DIR = "output"

# 모델 설정 (Hugging Face에서 자동 다운로드)
MODEL_ID = "iSEE-Laboratory/llmdet_large"  # 또는 "iSEE-Laboratory/llmdet_base", "iSEE-Laboratory/llmdet_large"
# 참고: 모델은 위에 설정된 CACHE_DIR (models/ 폴더)에 다운로드됩니다

# ReID 모델 설정 (BoTSORT 트래킹에 사용)
REID_WEIGHTS = Path(os.path.join(CACHE_DIR, "osnet_x0_25_msmt17.pt"))
# ================================================


def setup_model(model_id=MODEL_ID, device=None):
    """
    LLMDet 모델을 로드합니다 (Hugging Face에서 자동 다운로드).

    Args:
        model_id: Hugging Face 모델 ID
        device: 사용할 디바이스 (cuda/cpu)

    Returns:
        model: 로드된 LLMDet 모델
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


def setup_tracker(device):
    """
    BoTSORT 트래커를 초기화합니다.

    Args:
        device: 사용할 디바이스 (cuda/cpu)

    Returns:
        tracker: 초기화된 BoTSORT 트래커
    """
    print(f"BoTSORT 트래커 초기화 중... (ReID 활성화)")

    # boxmot은 device로 "cuda" 대신 GPU 번호 "0"을 기대함
    tracker_device = "0" if device == "cuda" else "cpu"

    # ID 스위칭 최소화를 위한 설정
    # - ReID 활성화: 외형 특징으로 같은 객체 인식
    # - new_track_thresh 높임: 새 트랙 생성 더 엄격하게
    # - match_thresh 낮춤: IoU 매칭 더 관대하게
    # - track_buffer 늘림: 트랙 유지 시간 증가
    tracker = BotSort(
        reid_weights=REID_WEIGHTS,
        device=tracker_device,
        half=False,
        with_reid=False,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=60,
        match_thresh=0.8,
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        cmc_method='sof',
        frame_rate=25
    )

    print(f"BoTSORT 트래커 초기화 완료! (ReID: {REID_WEIGHTS})")
    return tracker


def get_class_id_map(text_prompt):
    """
    텍스트 프롬프트를 파싱하여 고정된 클래스 ID 맵을 생성합니다.
    예: "car . person ." -> {"car": 0, "person": 1}
    """
    # 점(.)으로 구분하고 공백 제거, 빈 문자열 제거
    classes = [c.strip() for c in text_prompt.split('.') if c.strip()]
    return {c: i for i, c in enumerate(classes)}


def run_detection(model, processor, frame, text_prompt, class_id_map, box_threshold=0.35, text_threshold=0.25):
    """
    단일 프레임에 대해 LLMDet 추론을 실행합니다.

    Args:
        model: LLMDet 모델
        processor: 이미지 전처리 프로세서
        frame: 입력 프레임 (numpy array, BGR)
        text_prompt: 감지할 객체를 설명하는 텍스트 프롬프트
        class_id_map: 클래스 이름과 고정 ID 매핑 딕셔너리
        box_threshold: 바운딩 박스 신뢰도 임계값
        text_threshold: 텍스트 매칭 신뢰도 임계값

    Returns:
        detections: numpy array (N x 6) - [x1, y1, x2, y2, conf, cls]
    """
    # BGR to RGB 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)

    # 입력 전처리
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(model.device)

    # 추론
    with torch.no_grad():
        outputs = model(**inputs)

    # 후처리
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[image.size[::-1]]
    )[0]

    # threshold 적용하여 결과 필터링
    keep_indices = results['scores'] >= box_threshold
    
    # numpy 배열로 변환 (boxmot 입력 형식: [x1, y1, x2, y2, conf, cls])
    boxes = results['boxes'][keep_indices].cpu().numpy()
    scores = results['scores'][keep_indices].cpu().numpy()
    labels = [label for i, label in enumerate(results['labels']) if keep_indices[i]]
    
    # 각 레이블을 미리 생성한 고정 ID 맵을 사용하여 변환
    # 맵에 없는 레이블이 나올 경우(모델이 예측했지만 프롬프트 파싱과 다를 경우)를 대비해 해싱값 사용 등 예외처리 가능하나
    # 현재는 프롬프트 기반이므로 매칭됨. 매칭 안될 시 임시로 hash 사용.
    class_ids = []
    for label in labels:
        if label in class_id_map:
            class_ids.append(class_id_map[label])
        else:
            # 혹시 모를 예외 케이스: 해시값 등으로 고정
            class_ids.append(abs(hash(label)) % 100)
    class_ids = np.array(class_ids)
    
    # 감지 결과를 boxmot 형식으로 변환 [x1, y1, x2, y2, conf, cls]
    if len(boxes) > 0:
        detections = np.concatenate([
            boxes,
            scores[:, None],
            class_ids[:, None]
        ], axis=1)
    else:
        detections = np.empty((0, 6))
    
    return detections, labels


def get_color_for_id(track_id):
    """
    트래킹 ID에 따라 고유한 색상을 반환합니다.
    ID 스위칭 여부를 시각적으로 쉽게 파악할 수 있습니다.

    Args:
        track_id: 트래킹 ID

    Returns:
        color: BGR 색상 튜플
    """
    # 시각적으로 구분이 잘 되는 색상 팔레트 (BGR)
    colors = [
        (255, 0, 0),      # 파랑
        (0, 255, 0),      # 초록
        (0, 0, 255),      # 빨강
        (255, 255, 0),    # 시안
        (255, 0, 255),    # 마젠타
        (0, 255, 255),    # 노랑
        (128, 0, 255),    # 주황
        (255, 128, 0),    # 하늘색
        (128, 255, 0),    # 연두
        (0, 128, 255),    # 살구색
        (255, 0, 128),    # 보라
        (0, 255, 128),    # 청록
        (128, 128, 255),  # 연한 빨강
        (255, 128, 128),  # 연한 파랑
        (128, 255, 128),  # 연한 초록
        (128, 255, 255),  # 연한 노랑
    ]
    return colors[track_id % len(colors)]


def draw_tracking_results(frame, tracks):
    """
    프레임에 트래킹 결과를 그립니다.
    각 트래킹 ID마다 다른 색상을 사용하여 ID 스위칭을 쉽게 파악할 수 있습니다.

    Args:
        frame: 입력 프레임 (numpy array)
        tracks: 트래킹 결과 (N x 8) - [x1, y1, x2, y2, id, conf, cls, ind]

    Returns:
        frame: 어노테이션이 추가된 프레임
    """
    if tracks is None or len(tracks) == 0:
        return frame

    for track in tracks:
        x1, y1, x2, y2, track_id, conf, cls, ind = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)

        # ID별 고유 색상 가져오기
        color = get_color_for_id(track_id)

        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ID와 confidence 텍스트
        label = f"ID:{track_id} {conf:.2f}"

        # 텍스트 배경
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)

        # 텍스트 그리기
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame


def draw_detection_results(frame, detections, labels):
    """
    프레임에 객체 감지(ZSD) 결과를 그립니다.
    트래킹 ID 없이 클래스명, 점수, 바운딩 박스를 표시합니다.
    """
    if detections is None or len(detections) == 0:
        return frame

    for i, detection in enumerate(detections):
        x1, y1, x2, y2, score, cls_id = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # ID별 고유 색상 (클래스 ID 사용)
        color = get_color_for_id(int(cls_id))
        
        # 바운딩 박스
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 라벨 텍스트: Class Score
        label_text = labels[i]
        label = f"{label_text} {score:.2f}"
        
        # 텍스트 배경
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # 텍스트 그리기
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame


def process_video(video_path, output_path_tracking, output_path_detection, model, processor, tracker, text_prompt, box_threshold, text_threshold):
    """
    비디오를 처리하여 객체 감지 및 추적을 수행합니다.

    Args:
        video_path: 입력 비디오 경로
        output_path_tracking: 출력 트래킹 비디오 경로
        output_path_detection: 출력 감지(ZSD) 비디오 경로
        model: LLMDet 모델
        processor: 이미지 전처리 프로세서
        tracker: BoTSORT 트래커
        text_prompt: 감지할 객체를 설명하는 텍스트 프롬프트
        box_threshold: 바운딩 박스 신뢰도 임계값
        text_threshold: 텍스트 매칭 신뢰도 임계값
    """
    # 비디오 캡처 열기
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"오류: 비디오 파일을 열 수 없습니다: {video_path}")
        return
    
    # 비디오 정보 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"비디오 정보:")
    print(f"  - 해상도: {width}x{height}")
    print(f"  - FPS: {fps}")
    print(f"  - 총 프레임 수: {total_frames}")
    
    # 비디오 작성기 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_tracking = cv2.VideoWriter(output_path_tracking, fourcc, fps, (width, height))
    out_detection = cv2.VideoWriter(output_path_detection, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    print(f"\n비디오 처리 시작...")
    print(f"텍스트 프롬프트: '{text_prompt}'")
    
    # 클래스 ID 맵 생성 (고정된 색상을 위해)
    class_id_map = get_class_id_map(text_prompt)
    print(f"감지 클래스: {class_id_map}")
    print("-" * 60)
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # 감지 수행
            detections, labels = run_detection(
                model, processor, frame, text_prompt, class_id_map, box_threshold, text_threshold
            )
            
            # 1. Detection Only 결과 그리기 및 저장
            annotated_frame_detection = draw_detection_results(frame.copy(), detections, labels)
            
            info_text_det = f"Frame: {frame_count}/{total_frames} | Detections: {len(detections)}"
            cv2.putText(annotated_frame_detection, info_text_det, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            out_detection.write(annotated_frame_detection)
            
            # 2. 트래킹 업데이트 및 결과 그리기
            # INPUT: M X (x, y, x, y, conf, cls)
            # OUTPUT: M X (x, y, x, y, id, conf, cls, ind)
            tracks = tracker.update(detections, frame)
            
            annotated_frame_tracking = draw_tracking_results(frame.copy(), tracks)
            
            info_text_track = f"Frame: {frame_count}/{total_frames} | Objects: {len(tracks) if tracks is not None else 0}"
            cv2.putText(annotated_frame_tracking, info_text_track, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            out_tracking.write(annotated_frame_tracking)
            
            # 진행 상황 출력 (매 30프레임마다)
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps_current = frame_count / elapsed_time
                eta = (total_frames - frame_count) / fps_current if fps_current > 0 else 0
                print(f"처리 중: {frame_count}/{total_frames} 프레임 ({frame_count/total_frames*100:.1f}%) | "
                      f"FPS: {fps_current:.1f} | 남은 시간: {eta:.1f}초")
    
    except KeyboardInterrupt:
        print("\n처리가 사용자에 의해 중단되었습니다.")
    
    finally:
        # 정리
        cap.release()
        out_tracking.release()
        out_detection.release()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print("-" * 60)
        print(f"비디오 처리 완료!")
        print(f"  - 처리된 프레임 수: {frame_count}")
        print(f"  - 총 처리 시간: {total_time:.2f}초")
        print(f"  - 평균 FPS: {avg_fps:.1f}")
        print(f"  - 트래킹 영상: {output_path_tracking}")
        print(f"  - 감지 영상: {output_path_detection}")


def main():
    # 명령행 인자 파서 설정
    parser = argparse.ArgumentParser(
        description="LLMDet 비디오 추론 및 BoTSORT 트래킹 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python llmdet-video-tracking.py input_video.mp4
  python llmdet-video-tracking.py input_video.mp4 --prompt "person . car . bicycle ."
  python llmdet-video-tracking.py input_video.mp4 --box-threshold 0.4 --text-threshold 0.3
        """
    )
    
    parser.add_argument(
        "video",
        type=str,
        help="입력 비디오 파일 경로"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=TEXT_PROMPT,
        help=f"감지할 객체를 설명하는 텍스트 프롬프트 (기본값: '{TEXT_PROMPT}')"
    )
    
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=BOX_THRESHOLD,
        help=f"바운딩 박스 신뢰도 임계값 (기본값: {BOX_THRESHOLD})"
    )
    
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=TEXT_THRESHOLD,
        help=f"텍스트 매칭 신뢰도 임계값 (기본값: {TEXT_THRESHOLD})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"출력 비디오 저장 디렉토리 (기본값: {OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print(f"LLMDet 비디오 추론 및 BoTSORT 트래킹 시작")
    print("=" * 60)
    print(f"사용 디바이스: {device}")
    print(f"입력 비디오: {args.video}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"텍스트 프롬프트: '{args.prompt}'")
    print(f"Box threshold: {args.box_threshold}, Text threshold: {args.text_threshold}")
    print("=" * 60)
    
    # 입력 비디오 확인
    if not os.path.exists(args.video):
        print(f"\n오류: 비디오 파일을 찾을 수 없습니다: {args.video}")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 출력 파일명 생성
    video_basename = os.path.splitext(os.path.basename(args.video))[0]
    output_filename_tracking = f"{video_basename}_tracking.mp4"
    output_filename_detection = f"{video_basename}_detection.mp4"
    output_path_tracking = os.path.join(args.output_dir, output_filename_tracking)
    output_path_detection = os.path.join(args.output_dir, output_filename_detection)
    
    # 모델 로드
    print("\n" + "=" * 60)
    model, processor = setup_model(MODEL_ID, device)
    print("=" * 60)
    
    # 트래커 초기화
    print("\n" + "=" * 60)
    tracker = setup_tracker(device)
    print("=" * 60)
    
    # 비디오 처리
    print("\n" + "=" * 60)
    process_video(
        video_path=args.video,
        output_path_tracking=output_path_tracking,
        output_path_detection=output_path_detection,
        model=model,
        processor=processor,
        tracker=tracker,
        text_prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
