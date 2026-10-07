"""1단계: 제스처 데이터 수집 → gesture_data.csv

실행: python collect_gestures.py heart rock none   (이름은 생략 가능, 창에서 n으로도 추가)
  n          새 제스처 이름 입력 (영문 입력 후 Enter, 취소는 ESC)
  1~9        제스처 선택
  SPACE / REC 버튼 클릭
             1초 뒤 3초 동안 자동 녹화 (그동안 양손 자유)
  d          선택한 제스처의 데이터 삭제
  q / ESC    종료
"""
import csv
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
DATA_PATH = Path(__file__).parent / "gesture_data.csv"
CAMERA_INDEX = 0
WINDOW = "Collect Gestures"
COUNTDOWN_SEC = 1.0  # 녹화 버튼을 누르고 실제 녹화가 시작될 때까지
RECORD_SEC = 3.0     # 녹화 시간

# 21개 랜드마크 연결선 (엄지, 검지, 중지, 약지, 새끼, 손바닥)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def create_landmarker():
    options = vision.HandLandmarkerOptions(
        # 한글 경로는 MediaPipe가 못 열기 때문에 바이트로 읽어서 전달
        base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


def to_feature(hand_landmarks, handedness, w, h):
    """손 위치·크기와 무관하게 손 모양만 남긴 63개 숫자. (추론 때도 같은 함수를 씀)"""
    pts = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in hand_landmarks], dtype=np.float32)
    pts -= pts[0]                      # 손목을 원점으로
    if handedness == "Left":
        pts[:, 0] *= -1                # 왼손은 좌우 반전 → 오른손과 같게 취급
    pts /= max(np.linalg.norm(pts[:, :2], axis=1).max(), 1e-6)  # 손 크기 맞추기
    return pts.reshape(-1)


def draw_hand(frame, hand_landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)
    return pts


def load_data():
    rows = []
    if DATA_PATH.exists():
        with DATA_PATH.open(newline="", encoding="utf-8") as f:
            rows = [row for row in csv.reader(f) if row]
    return rows  # [라벨, 숫자 63개]


def save_data(rows):
    with DATA_PATH.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def main():
    rows = load_data()
    labels = list(dict.fromkeys([*(row[0] for row in rows), *sys.argv[1:]]))

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. CAMERA_INDEX를 확인하세요.")

    clicks = []
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, lambda e, x, y, *_: e == cv2.EVENT_LBUTTONDOWN and clicks.append((x, y)))

    current = 0        # 선택된 제스처 번호
    rec_start = None   # 녹화 버튼을 누른 시각
    typing = None      # 새 이름 입력 중이면 문자열
    start = time.monotonic()
    with create_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드
            h, w = frame.shape[:2]
            now = time.monotonic()
            button = (w - 160, h - 80, w - 20, h - 20)  # REC 버튼 (오른쪽 아래)

            # REC 버튼 클릭
            for cx, cy in clicks:
                if button[0] <= cx <= button[2] and button[1] <= cy <= button[3] \
                        and labels and rec_start is None:
                    rec_start = now
                    print(f"[{labels[current]}] 녹화 준비")
            clicks.clear()

            # 녹화 상태: 대기 → 녹화 → 끝나면 저장
            phase = None
            if rec_start is not None:
                t = now - rec_start
                if t < COUNTDOWN_SEC:
                    phase = "ready"
                elif t < COUNTDOWN_SEC + RECORD_SEC:
                    phase = "rec"
                else:
                    rec_start = None
                    save_data(rows)
                    n = sum(row[0] == labels[current] for row in rows)
                    print(f"[{labels[current]}] 녹화 끝 → 총 {n}개 저장")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int((now - start) * 1000))

            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                draw_hand(frame, hand_landmarks)
                if phase == "rec":
                    feat = to_feature(hand_landmarks, handedness[0].category_name, w, h)
                    rows.append([labels[current], *(f"{v:.4f}" for v in feat)])

            # 왼쪽 위: 제스처 목록과 개수
            counts = Counter(row[0] for row in rows)
            for i, label in enumerate(labels):
                selected = i == current
                color = (0, 0, 255) if selected and phase else (0, 255, 255) if selected else (200, 200, 200)
                cv2.putText(frame, f"{'>' if selected else ' '}{i + 1}: {label} ({counts[label]})",
                            (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            msg = f"new name: {typing}_  (Enter)" if typing is not None \
                else "" if labels else "press n to add a gesture name (English input mode)"
            cv2.putText(frame, msg, (10, 30 + len(labels) * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 대기 / 녹화 표시
            if phase == "ready":
                cv2.putText(frame, "READY", (w // 2 - 90, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            elif phase == "rec":
                left = COUNTDOWN_SEC + RECORD_SEC - (now - rec_start)
                cv2.putText(frame, f"REC {left:.1f}s  {labels[current]}", (10, h - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                if not result.hand_landmarks:
                    cv2.putText(frame, "NO HAND", (w // 2 - 100, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # REC 버튼
            cv2.rectangle(frame, button[:2], button[2:], (0, 0, 255) if phase else (80, 80, 80), -1)
            cv2.circle(frame, (button[0] + 30, h - 50), 10, (255, 255, 255), -1)
            cv2.putText(frame, "REC", (button[0] + 55, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
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
            elif rec_start is not None:
                continue  # 녹화 중에는 다른 키 무시
            elif key == ord("n"):
                typing = ""
            elif key == ord(" ") and labels:
                rec_start = now
                print(f"[{labels[current]}] 녹화 준비")
            elif ord("1") <= key < ord("1") + min(len(labels), 9):
                current = key - ord("1")
            elif key == ord("d") and labels:
                rows = [row for row in rows if row[0] != labels[current]]
                save_data(rows)
                print(f"[{labels[current]}] 데이터 삭제")

    cap.release()
    cv2.destroyAllWindows()
    save_data(rows)
    print(f"저장: {DATA_PATH}  {dict(Counter(row[0] for row in rows))}")


if __name__ == "__main__":
    main()
