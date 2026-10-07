"""Hand Landmarker + Gesture Recognizer 동시 실행 (웹캠 1대 공유).

왼쪽 화면: Hand Landmarker (랜드마크 + 왼손/오른손)
오른쪽 화면: Gesture Recognizer (제스처 이름 + 점수)
실행: python hand_gesture_webcam.py   (종료: q 또는 ESC)
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from gesture_webcam import draw_result as draw_gestures
from hand_webcam import draw_hands

BASE_DIR = Path(__file__).parent
CAMERA_INDEX = 0
NUM_HANDS = 2


def base_options(name):
    # 한글 경로는 MediaPipe가 못 열기 때문에 바이트로 읽어서 전달
    return mp_python.BaseOptions(model_asset_buffer=(BASE_DIR / name).read_bytes())


def main():
    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options("hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
    )
    gesture_options = vision.GestureRecognizerOptions(
        base_options=base_options("gesture_recognizer.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. 다른 프로그램이 사용 중인지 확인하세요.")

    start = time.monotonic()
    prev = start
    with vision.HandLandmarker.create_from_options(hand_options) as landmarker, \
            vision.GestureRecognizer.create_from_options(gesture_options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            hand_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            gesture_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            hand_view = frame.copy()
            gesture_view = frame.copy()
            draw_hands(hand_view, hand_result)
            draw_gestures(gesture_view, gesture_result)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(hand_view, f"Hand Landmarker  FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(gesture_view, "Gesture Recognizer", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Hand Landmarker | Gesture Recognizer",
                       np.hstack([hand_view, gesture_view]))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
