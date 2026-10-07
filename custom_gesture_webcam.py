"""3단계: 학습한 모델로 실시간 제스처 인식 + heart 제스처를 하면 하트 이모지가 날아감

실행: python custom_gesture_webcam.py   (종료: q 또는 ESC)
확률이 THRESHOLD보다 낮으면 "?"로 표시한다.
"""
import math
import random
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from collect_gestures import CAMERA_INDEX, create_landmarker, draw_hand, to_feature
from train_gestures import MODEL_PATH, build_model

THRESHOLD = 0.7
EFFECT_GESTURE = "heart"  # 이 제스처를 하면 하트가 날아감
EMOJI_FONT = "C:/Windows/Fonts/seguiemj.ttf"
SPAWN_PER_SEC = 15        # 초당 생기는 하트 수


def make_heart_sprite():
    """하트 이모지를 투명 배경 BGRA 이미지로 만든다 (cv2.putText는 이모지를 못 그림)."""
    img = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(EMOJI_FONT, 109)
        draw.text((50, 50), "\u2764\ufe0f", font=font, embedded_color=True)
    except OSError:  # 이모지 폰트가 없으면 직접 그린 하트
        draw.ellipse((50, 50, 150, 150), fill=(230, 30, 60, 255))
        draw.ellipse((130, 50, 230, 150), fill=(230, 30, 60, 255))
        draw.polygon([(55, 120), (225, 120), (140, 230)], fill=(230, 30, 60, 255))
    img = img.crop(img.getbbox())
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def overlay(frame, sprite, cx, cy, size, alpha):
    """frame의 (cx, cy)를 중심으로 sprite를 size 크기, alpha 투명도로 합성."""
    sh, sw = sprite.shape[:2]
    w, h = max(int(size), 1), max(int(size * sh / sw), 1)
    img = cv2.resize(sprite, (w, h), interpolation=cv2.INTER_AREA)
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    fx1, fy1 = max(x1, 0), max(y1, 0)
    fx2, fy2 = min(x1 + w, frame.shape[1]), min(y1 + h, frame.shape[0])
    if fx1 >= fx2 or fy1 >= fy2:
        return  # 화면 밖
    part = img[fy1 - y1:fy2 - y1, fx1 - x1:fx2 - x1]
    a = part[..., 3:4].astype(np.float32) / 255 * alpha
    roi = frame[fy1:fy2, fx1:fx2]
    roi[:] = (part[..., :3] * a + roi * (1 - a)).astype(np.uint8)


class Hearts:
    """손에서 생겨 위로 흔들리며 날아가다 사라지는 하트들."""

    def __init__(self):
        self.sprite = make_heart_sprite()
        self.items = []

    def spawn(self, x, y, now):
        self.items.append({
            "x": x, "y": y, "born": now,
            "vx": random.uniform(-120, 120),   # 픽셀/초
            "vy": random.uniform(-380, -180),  # 위로
            "size": random.uniform(35, 80),
            "life": random.uniform(1.2, 2.2),  # 초
            "phase": random.uniform(0, 2 * math.pi),
        })

    def update_and_draw(self, frame, now, dt):
        alive = []
        for p in self.items:
            age = (now - p["born"]) / p["life"]  # 0 → 1
            if age >= 1:
                continue
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            wobble = math.sin(now * 6 + p["phase"]) * 15      # 좌우로 살랑살랑
            size = p["size"] * (0.5 + 0.7 * min(age * 3, 1))  # 처음엔 작다가 커짐
            alpha = 1.0 if age < 0.6 else (1 - age) / 0.4     # 끝날 때 서서히 사라짐
            overlay(frame, self.sprite, p["x"] + wobble, p["y"], size, alpha)
            alive.append(p)
        self.items = alive


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"모델이 없습니다: {MODEL_PATH}  (먼저 train_gestures.py 실행)")
    ckpt = torch.load(MODEL_PATH, weights_only=True)
    labels = ckpt["labels"]
    model = build_model(len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print("제스처:", labels)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. CAMERA_INDEX를 확인하세요.")

    hearts = Hearts()
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

            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                pts = draw_hand(frame, hand_landmarks)
                feat = to_feature(hand_landmarks, handedness[0].category_name, w, h)
                with torch.no_grad():
                    probs = torch.softmax(model(torch.from_numpy(feat)[None]), dim=1)[0]
                score, idx = probs.max(0)
                name = labels[int(idx)] if score >= THRESHOLD else "?"

                if name == EFFECT_GESTURE and random.random() < SPAWN_PER_SEC * dt:
                    cx = sum(p[0] for p in pts) / len(pts)
                    cy = min(p[1] for p in pts)  # 손 위쪽에서 출발
                    hearts.spawn(cx, cy, now)

                x0 = min(p[0] for p in pts)
                y0 = max(min(p[1] for p in pts) - 10, 30)
                cv2.putText(frame, f"{name} {score:.2f}", (x0, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

            hearts.update_and_draw(frame, now, dt)

            cv2.imshow("Custom Gesture", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
