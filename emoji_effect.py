"""이모지 그리기 + 날아가는 이모지 효과 (collect_gestures.py, custom_gesture_webcam.py 공용).

제스처 이름 → 이모지 연결은 gesture_emojis.json 에 저장된다.
"""
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EMOJI_FONT = "C:/Windows/Fonts/seguiemj.ttf"  # Windows 기본 컬러 이모지 폰트
EMOJI_PATH = Path(__file__).parent / "gesture_emojis.json"
DEFAULT_EMOJIS = {"heart": "\u2764\ufe0f"}  # ❤️

# 수집 화면에서 고를 수 있는 이모지 (2열 x 8행)
PALETTE = [
    "\u2764\ufe0f", "\u270c\ufe0f", "\U0001F44C", "\U0001F44D",  # ❤️ ✌️ 👌 👍
    "\U0001F44E", "\U0001F44F", "\U0001F44B", "\u270a",          # 👎 👏 👋 ✊
    "\U0001F91F", "\U0001F64F", "\U0001F525", "\u2b50",          # 🤟 🙏 🔥 ⭐
    "\U0001F389", "\U0001F600", "\U0001F60D", "\U0001F4AF",      # 🎉 😀 😍 💯
]

_sprites = {}


def load_emojis():
    if EMOJI_PATH.exists():
        return json.loads(EMOJI_PATH.read_text(encoding="utf-8"))
    return dict(DEFAULT_EMOJIS)


def save_emojis(emojis):
    EMOJI_PATH.write_text(json.dumps(emojis, ensure_ascii=False, indent=2), encoding="utf-8")


def emoji_sprite(emoji):
    """이모지를 투명 배경 BGRA 이미지로 (cv2.putText는 이모지를 못 그림)."""
    if emoji not in _sprites:
        img = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
        try:
            font = ImageFont.truetype(EMOJI_FONT, 109)
            ImageDraw.Draw(img).text((50, 50), emoji, font=font, embedded_color=True)
        except OSError:
            pass
        if img.getbbox() is None:  # 폰트가 없으면 빨간 동그라미로 대신
            ImageDraw.Draw(img).ellipse((50, 50, 150, 150), fill=(230, 30, 60, 255))
        img = img.crop(img.getbbox())
        _sprites[emoji] = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    return _sprites[emoji]


def overlay(frame, emoji, cx, cy, size, alpha=1.0):
    """frame의 (cx, cy)를 중심으로 이모지를 size 크기(긴 쪽 픽셀), alpha 투명도로 그린다."""
    sprite = emoji_sprite(emoji)
    sh, sw = sprite.shape[:2]
    scale = size / max(sh, sw)
    w, h = max(int(sw * scale), 1), max(int(sh * scale), 1)
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


class FlyingEmojis:
    """손에서 생겨 위로 흔들리며 날아가다 사라지는 이모지들."""

    def __init__(self):
        self.items = []

    def spawn(self, emoji, x, y, now):
        self.items.append({
            "emoji": emoji, "x": x, "y": y, "born": now,
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
            overlay(frame, p["emoji"], p["x"] + wobble, p["y"], size, alpha)
            alive.append(p)
        self.items = alive
