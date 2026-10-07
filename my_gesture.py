"""나만의 제스처 인식 - 수집과 인식을 한 파일로 (따로 학습할 필요 없음).

실행: python my_gesture.py
  n          새 제스처 이름 입력 (영문 입력 후 Enter, 취소는 ESC)
  1~9 / 클릭  제스처 선택
  SPACE / REC 버튼 클릭
             1초 뒤 3초 동안 자동 녹화 → 그동안 양손을 자유롭게 쓸 수 있음
  d          선택한 제스처의 데이터 삭제
  q / ESC    종료
화면에는 저장한 샘플 중 가장 비슷한 제스처가 바로 표시된다.
데이터는 my_gestures.npz 에 저장되어 다음 실행 때 이어서 쓴다.
"""
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
DATA_PATH = Path(__file__).parent / "my_gestures.npz"
CAMERA_INDEX = 0
WINDOW = "My Gesture"
COUNTDOWN_SEC = 1.0  # 녹화 버튼을 누르고 실제 녹화가 시작될 때까지
RECORD_SEC = 3.0     # 녹화 시간
K = 5                # 가장 가까운 샘플 K개로 투표
MAX_DIST = 0.5       # 가장 가까운 샘플이 이보다 멀면 "?" (모르는 손 모양)
ROW_H = 30           # 왼쪽 위 제스처 목록 한 줄 높이

# 21개 랜드마크 연결선 (엄지, 검지, 중지, 약지, 새끼, 손바닥)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def to_feature(hand_landmarks, handedness, w, h):
    """손 위치·크기와 무관하게 손 모양만 남긴 63개 숫자."""
    pts = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in hand_landmarks], dtype=np.float32)
    pts -= pts[0]                      # 손목을 원점으로
    if handedness == "Left":
        pts[:, 0] *= -1                # 왼손은 좌우 반전 → 오른손과 같게 취급
    pts /= max(np.linalg.norm(pts[:, :2], axis=1).max(), 1e-6)  # 손 크기 맞추기
    return pts.reshape(-1)


def predict(feat, X, y):
    if len(X) == 0:
        return "", 0.0
    dist = np.linalg.norm(X - feat, axis=1)
    nearest = np.argsort(dist)[:K]
    if dist[nearest[0]] > MAX_DIST:
        return "?", dist[nearest[0]]
    return Counter(y[nearest]).most_common(1)[0][0], dist[nearest[0]]


def rec_button(w, h):
    return w - 160, h - 80, w - 20, h - 20  # x1, y1, x2, y2 (오른쪽 아래)


def main():
    X, y = np.zeros((0, 63), np.float32), np.array([], dtype=str)
    if DATA_PATH.exists():
        data = np.load(DATA_PATH)
        X, y = data["X"], data["y"]
    labels = list(dict.fromkeys([*y.tolist(), *sys.argv[1:]]))

    def save():
        np.savez(DATA_PATH, X=X, y=y)

    options = vision.HandLandmarkerOptions(
        # 한글 경로는 MediaPipe가 못 열기 때문에 바이트로 읽어서 전달
        base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. CAMERA_INDEX를 확인하세요.")

    clicks = []
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, lambda e, x, y_, *_: e == cv2.EVENT_LBUTTONDOWN and clicks.append((x, y_)))

    current = 0        # 선택된 제스처 번호
    rec_start = None   # 녹화 버튼을 누른 시각
    typing = None      # 새 이름 입력 중이면 문자열
    start = time.monotonic()
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드
            h, w = frame.shape[:2]
            now = time.monotonic()

            # 마우스 클릭: REC 버튼 또는 제스처 목록
            bx1, by1, bx2, by2 = rec_button(w, h)
            for cx, cy in clicks:
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    if labels and rec_start is None:
                        rec_start = now
                elif cx < 300 and 0 <= (cy - 8) // ROW_H < len(labels) and rec_start is None:
                    current = (cy - 8) // ROW_H
            clicks.clear()

            # 녹화 상태: 카운트다운 → 녹화 → 끝나면 저장
            phase = None
            if rec_start is not None:
                t = now - rec_start
                if t < COUNTDOWN_SEC:
                    phase = "ready"
                elif t < COUNTDOWN_SEC + RECORD_SEC:
                    phase = "rec"
                else:
                    rec_start = None
                    save()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int((now - start) * 1000))

            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                feat = to_feature(hand_landmarks, handedness[0].category_name, w, h)
                if phase == "rec":
                    X = np.vstack([X, feat])
                    y = np.append(y, labels[current])

                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                for p in pts:
                    cv2.circle(frame, p, 4, (0, 0, 255), -1)

                name, dist = predict(feat, X, y)
                x0 = min(p[0] for p in pts)
                y0 = max(min(p[1] for p in pts) - 10, 30)
                cv2.putText(frame, f"{name} ({dist:.2f})", (x0, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

            # 왼쪽 위: 제스처 목록 (선택된 것은 노란색, 녹화 중이면 빨간색)
            counts = Counter(y.tolist())
            for i, label in enumerate(labels):
                selected = i == current
                color = (0, 0, 255) if selected and phase else (0, 255, 255) if selected else (200, 200, 200)
                cv2.putText(frame, f"{'>' if selected else ' '}{i + 1}: {label} ({counts[label]})",
                            (10, 30 + i * ROW_H), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            row_y = 30 + len(labels) * ROW_H
            if typing is not None:
                cv2.putText(frame, f"new name: {typing}_  (Enter)", (10, row_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif not labels:
                cv2.putText(frame, "press n to add a gesture name", (10, row_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 가운데: 카운트다운 / 녹화 남은 시간
            if phase == "ready":
                cv2.putText(frame, "READY", (w // 2 - 90, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            elif phase == "rec":
                left = COUNTDOWN_SEC + RECORD_SEC - (now - rec_start)
                cv2.putText(frame, f"REC {left:.1f}s  {labels[current]}", (10, h - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # 오른쪽 아래: REC 버튼
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255) if phase else (80, 80, 80), -1)
            cv2.circle(frame, (bx1 + 30, (by1 + by2) // 2), 10, (255, 255, 255), -1)
            cv2.putText(frame, "REC", (bx1 + 55, by2 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, "n: new  1-9: select  SPACE: rec  d: delete  q: quit",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if typing is not None:  # 이름 입력 중에는 키를 글자로 받음
                if key == 13:  # Enter
                    if typing and typing not in labels:
                        labels.append(typing)
                    if typing in labels:
                        current = labels.index(typing)
                    typing = None
                elif key == 27:  # ESC: 취소
                    typing = None
                elif key == 8:  # Backspace
                    typing = typing[:-1]
                elif 33 <= key <= 126:
                    typing += chr(key)
            elif key in (ord("q"), 27):
                break
            elif key == ord("n") and rec_start is None:
                typing = ""
            elif key == ord(" ") and labels and rec_start is None:
                rec_start = now
            elif ord("1") <= key < ord("1") + min(len(labels), 9) and rec_start is None:
                current = key - ord("1")
            elif key == ord("d") and labels and rec_start is None:
                keep = y != labels[current]
                X, y = X[keep], y[keep]
                save()

    cap.release()
    cv2.destroyAllWindows()
    save()
    print(f"저장: {DATA_PATH}  {dict(Counter(y.tolist()))}")


if __name__ == "__main__":
    main()
