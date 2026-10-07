"""MediaPipe Gesture Recognizer - 웹캠 실시간 손 제스처 인식.

인식 제스처: None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down,
            Thumb_Up, Victory, ILoveYou
실행: python gesture_webcam.py   (종료: q 또는 ESC)
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "gesture_recognizer.task"
CAMERA_INDEX = 0
NUM_HANDS = 2

# 21개 랜드마크 연결선 (엄지, 검지, 중지, 약지, 새끼, 손바닥)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_result(frame, result):
    h, w = frame.shape[:2]
    for hand_landmarks, handedness, gestures in zip(
            result.hand_landmarks, result.handedness, result.gestures):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # 제스처 이름 + 점수, 손 라벨 (Left/Right)
        top = gestures[0]
        x0 = min(p[0] for p in pts)
        y0 = max(min(p[1] for p in pts) - 10, 50)
        cv2.putText(frame, f"{top.category_name} {top.score:.2f}", (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        cv2.putText(frame, handedness[0].category_name, (x0, y0 - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    options = vision.GestureRecognizerOptions(
        # 한글 경로는 MediaPipe가 못 열기 때문에 바이트로 읽어서 전달
        base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. CAMERA_INDEX를 확인하세요.")

    start = time.monotonic()
    prev = start
    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            draw_result(frame, result)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}  Hands: {len(result.hand_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("MediaPipe Gesture Recognizer", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
