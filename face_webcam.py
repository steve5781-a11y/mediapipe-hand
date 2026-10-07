"""MediaPipe Face Landmarker - 웹캠 실시간 얼굴 랜드마크(478점) + 표정 blendshape.

실행: python face_webcam.py   (종료: q 또는 ESC, m: 메쉬 표시 토글)
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "face_landmarker.task"
CAMERA_INDEX = 0
NUM_FACES = 1
TOP_BLENDSHAPES = 8  # 화면에 표시할 상위 표정 개수

C = vision.FaceLandmarksConnections
CONTOUR_GROUPS = [
    (C.FACE_LANDMARKS_FACE_OVAL, (224, 224, 224)),
    (C.FACE_LANDMARKS_LEFT_EYE, (48, 255, 48)),
    (C.FACE_LANDMARKS_LEFT_EYEBROW, (48, 255, 48)),
    (C.FACE_LANDMARKS_RIGHT_EYE, (48, 48, 255)),
    (C.FACE_LANDMARKS_RIGHT_EYEBROW, (48, 48, 255)),
    (C.FACE_LANDMARKS_LIPS, (180, 105, 255)),
    (C.FACE_LANDMARKS_LEFT_IRIS, (255, 255, 0)),
    (C.FACE_LANDMARKS_RIGHT_IRIS, (255, 255, 0)),
]


def draw_result(frame, result, show_mesh):
    h, w = frame.shape[:2]
    for face_landmarks in result.face_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
        if show_mesh:
            for c in C.FACE_LANDMARKS_TESSELATION:
                cv2.line(frame, pts[c.start], pts[c.end], (90, 90, 90), 1)
        for connections, color in CONTOUR_GROUPS:
            for c in connections:
                cv2.line(frame, pts[c.start], pts[c.end], color, 1)

    # 첫 번째 얼굴의 blendshape 상위 N개를 막대로 표시
    if result.face_blendshapes:
        shapes = sorted(result.face_blendshapes[0], key=lambda s: s.score, reverse=True)
        for i, s in enumerate(shapes[:TOP_BLENDSHAPES]):
            y = 60 + i * 24
            cv2.rectangle(frame, (10, y - 14), (10 + int(s.score * 150), y + 4), (0, 200, 255), -1)
            cv2.putText(frame, f"{s.category_name} {s.score:.2f}", (170, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    options = vision.FaceLandmarkerOptions(
        # 한글 경로는 MediaPipe가 못 열기 때문에 바이트로 읽어서 전달
        base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=NUM_FACES,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. CAMERA_INDEX를 확인하세요.")

    show_mesh = True
    start = time.monotonic()
    prev = start
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_result(frame, result, show_mesh)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}  Faces: {len(result.face_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("MediaPipe Face Landmarker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                show_mesh = not show_mesh

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
