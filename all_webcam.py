"""Hand Landmarker + Gesture Recognizer + Face Landmarker 동시 실행 (웹캠 1대 공유).

한 화면에 세 결과를 겹쳐서 표시:
  - Face Landmarker: 얼굴 메쉬 + 윤곽, 왼쪽 위 표정(blendshape) 점수
  - Hand Landmarker: 손 랜드마크 + 왼손/오른손
  - Gesture Recognizer: 손 위에 제스처 이름 + 점수
실행: python all_webcam.py   (종료: q 또는 ESC, m: 얼굴 메쉬 표시 토글)
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from face_webcam import draw_result as draw_face
from hand_webcam import draw_hands

BASE_DIR = Path(__file__).parent
CAMERA_INDEX = 0
NUM_HANDS = 2
NUM_FACES = 1


def base_options(name):
    # 한글 경로는 MediaPipe가 못 열기 때문에 바이트로 읽어서 전달
    return mp_python.BaseOptions(model_asset_buffer=(BASE_DIR / name).read_bytes())


def draw_gesture_labels(frame, result):
    # 손 골격은 Hand Landmarker가 그리므로 여기서는 제스처 이름만 손 위에 표시
    h, w = frame.shape[:2]
    for hand_landmarks, gestures in zip(result.hand_landmarks, result.gestures):
        top = gestures[0]
        x0 = int(min(lm.x for lm in hand_landmarks) * w)
        y0 = int(min(lm.y for lm in hand_landmarks) * h) - 38
        cv2.putText(frame, f"{top.category_name} {top.score:.2f}", (x0, max(y0, 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)


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
    face_options = vision.FaceLandmarkerOptions(
        base_options=base_options("face_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=NUM_FACES,
        output_face_blendshapes=True,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. 다른 프로그램이 사용 중인지 확인하세요.")

    show_mesh = True
    start = time.monotonic()
    prev = start
    with vision.HandLandmarker.create_from_options(hand_options) as landmarker, \
            vision.GestureRecognizer.create_from_options(gesture_options) as recognizer, \
            vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:
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
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_face(frame, face_result, show_mesh)
            draw_hands(frame, hand_result)
            draw_gesture_labels(frame, gesture_result)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}  Hands: {len(hand_result.hand_landmarks)}"
                        f"  Faces: {len(face_result.face_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Hand + Gesture + Face", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                show_mesh = not show_mesh

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
