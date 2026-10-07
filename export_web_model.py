"""4단계(선택): 학습한 모델을 웹 데모용 JSON으로 변환 → docs/gesture_model.json

실행: python export_web_model.py
docs/ 폴더가 GitHub Pages로 배포되므로, 다시 학습했으면 이걸 실행하고 커밋/푸시하면 웹에도 반영된다.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from emoji_effect import load_emojis
from train_gestures import DATA_PATH, MODEL_PATH, build_model

OUT_PATH = Path(__file__).parent / "docs" / "gesture_model.json"


def main():
    ckpt = torch.load(MODEL_PATH, weights_only=True)
    labels = ckpt["labels"]
    model = build_model(len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # nn.Sequential 안의 Linear 층만 순서대로 (weight: [출력, 입력])
    layers = [{"w": np.round(m.weight.detach().numpy(), 6).tolist(),
               "b": np.round(m.bias.detach().numpy(), 6).tolist()}
              for m in model if isinstance(m, torch.nn.Linear)]
    emojis = load_emojis()
    OUT_PATH.write_text(json.dumps({
        "labels": labels,
        "emojis": {label: emojis[label] for label in labels if label in emojis},
        "layers": layers,
    }, ensure_ascii=False), encoding="utf-8")

    # 확인: JSON 가중치로 직접 계산한 결과가 PyTorch와 같은지
    if DATA_PATH.exists():
        with DATA_PATH.open(newline="", encoding="utf-8") as f:
            X = np.array([row[1:] for row in csv.reader(f) if row], dtype=np.float32)
        a = X
        for i, layer in enumerate(layers):
            a = a @ np.array(layer["w"]).T + np.array(layer["b"])
            if i < len(layers) - 1:
                a = np.maximum(a, 0)
        with torch.no_grad():
            ref = model(torch.from_numpy(X)).argmax(1).numpy()
        print(f"검증: 샘플 {len(X)}개 중 PyTorch와 예측이 같은 비율 {(a.argmax(1) == ref).mean():.1%}")

    print(f"저장: {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)  제스처: {labels}")


if __name__ == "__main__":
    main()
