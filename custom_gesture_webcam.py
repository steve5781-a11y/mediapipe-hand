"""3단계: 학습한 모델로 실시간 제스처 인식 + 이모지 효과

실행: python custom_gesture_webcam.py   (종료: q 또는 ESC)
- 확률이 THRESHOLD보다 낮으면 "?"로 표시한다.
- 이모지가 연결된 제스처(collect_gestures.py에서 선택)를 하면
  화면 위에 "인식됨" 표시가 뜨고 손에서 그 이모지가 날아간다.
- 오른쪽 패널: 학습한 제스처 목록 + 확률 막대, 지금 인식된 제스처는 초록색
"""
import random
import time

import cv2
import mediapipe as mp
import numpy as np
import torch

from collect_gestures import CAMERA_INDEX, create_landmarker, draw_hand, to_feature
from emoji_effect import FlyingEmojis, load_emojis, overlay
from train_gestures import MODEL_PATH, build_model

THRESHOLD = 0.7
SPAWN_PER_SEC = 15  # 초당 생기는 이모지 수
PANEL_W = 260       # 오른쪽 제스처 목록 패널 너비


def draw_banner(frame, name, emoji):
    """화면 위쪽 가운데에 '이모지 + 제스처 이름 detected!' 표시."""
    w = frame.shape[1]
    text = f"{name} detected!"
    tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0][0]
    x = (w - tw - 70) // 2
    cv2.rectangle(frame, (x - 15, 10), (x + tw + 85, 80), (40, 40, 40), -1)
    overlay(frame, emoji, x + 25, 45, 50)
    cv2.putText(frame, text, (x + 65, 57), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)


def draw_panel(h, labels, emojis, probs, recognized):
    """학습한 제스처 목록 + 각 확률 막대. 인식된 제스처는 초록 배경으로 강조."""
    panel = np.full((h, PANEL_W, 3), 30, np.uint8)
    cv2.putText(panel, "Gestures", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    row_h = min(60, (h - 140) // max(len(labels), 1))
    for i, (label, p) in enumerate(zip(labels, probs)):
        y = 55 + i * row_h
        if label in recognized:
            cv2.rectangle(panel, (5, y), (PANEL_W - 5, y + row_h - 6), (0, 130, 0), -1)
        if label in emojis:
            overlay(panel, emojis[label], 32, y + (row_h - 6) // 2, row_h - 22)
        cv2.putText(panel, label, (62, y + row_h // 2 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        bar_y = y + row_h // 2 + 6
        cv2.rectangle(panel, (62, bar_y), (PANEL_W - 60, bar_y + 8), (80, 80, 80), -1)
        if int((PANEL_W - 122) * p) > 0:
            cv2.rectangle(panel, (62, bar_y), (62 + int((PANEL_W - 122) * p), bar_y + 8), (0, 220, 255), -1)
        cv2.putText(panel, f"{p:.0%}", (PANEL_W - 52, bar_y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    # 맨 아래: 지금 인식된 제스처
    cv2.line(panel, (10, h - 75), (PANEL_W - 10, h - 75), (90, 90, 90), 1)
    cv2.putText(panel, "Now:", (15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    if recognized:
        name = recognized[0]
        if name in emojis:
            overlay(panel, emojis[name], 105, h - 38, 44)
        cv2.putText(panel, name, (135, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(panel, "-", (95, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)
    return panel


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"모델이 없습니다: {MODEL_PATH}  (먼저 train_gestures.py 실행)")
    ckpt = torch.load(MODEL_PATH, weights_only=True)
    labels = ckpt["labels"]
    model = build_model(len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    emojis = load_emojis()
    print("제스처:", labels, " 이모지 연결됨:", [l for l in labels if l in emojis])

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. CAMERA_INDEX를 확인하세요.")

    cv2.namedWindow("Custom Gesture")
    cv2.setWindowProperty("Custom Gesture", cv2.WND_PROP_TOPMOST, 1)  # 다른 창에 가려지지 않게 맨 앞에
    flying = FlyingEmojis()
    start = prev = time.monotonic()
    with create_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드 (수집 때와 같아야 함)
            h, w = frame.shape[:2]
            now = time.monotonic()
            dt, prev = now - prev, now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int((now - start) * 1000))

            banner = None
            best = np.zeros(len(labels))  # 손이 여러 개면 제스처별 가장 높은 확률
            recognized = []               # 지금 인식된 제스처 이름들
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                pts = draw_hand(frame, hand_landmarks)
                feat = to_feature(hand_landmarks, handedness[0].category_name, w, h)
                with torch.no_grad():
                    probs = torch.softmax(model(torch.from_numpy(feat)[None]), dim=1)[0]
                best = np.maximum(best, probs.numpy())
                score, idx = probs.max(0)
                name = labels[int(idx)] if score >= THRESHOLD else "?"
                if name != "?" and name not in recognized:
                    recognized.append(name)

                if name in emojis:
                    banner = banner or (name, emojis[name])
                    if random.random() < SPAWN_PER_SEC * dt:
                        cx = sum(p[0] for p in pts) / len(pts)
                        cy = min(p[1] for p in pts)  # 손 위쪽에서 출발
                        flying.spawn(emojis[name], cx, cy, now)

                x0 = min(p[0] for p in pts)
                y0 = max(min(p[1] for p in pts) - 10, 30)
                cv2.putText(frame, f"{name} {score:.2f}", (x0, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

            flying.update_and_draw(frame, now, dt)
            if banner:
                draw_banner(frame, *banner)

            frame = np.hstack([frame, draw_panel(h, labels, emojis, best, recognized)])
            cv2.imshow("Custom Gesture", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
