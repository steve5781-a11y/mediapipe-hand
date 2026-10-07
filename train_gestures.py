"""2단계: 모은 데이터로 제스처 분류 모델 학습 → gesture_model.pt

실행: python train_gestures.py
"""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

DATA_PATH = Path(__file__).parent / "gesture_data.csv"
MODEL_PATH = Path(__file__).parent / "gesture_model.pt"
EPOCHS = 300


def build_model(num_classes):
    """손 모양 숫자 63개 → 제스처별 점수"""
    return nn.Sequential(
        nn.Linear(63, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, num_classes),
    )


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"데이터가 없습니다: {DATA_PATH}  (먼저 collect_gestures.py 실행)")
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.reader(f) if row]

    labels = sorted({row[0] for row in rows})
    print("데이터:", dict(Counter(row[0] for row in rows)))
    if len(labels) < 2:
        raise SystemExit("제스처가 2개 이상 있어야 학습할 수 있습니다.")

    X = torch.tensor(np.array([row[1:] for row in rows], dtype=np.float32))
    y = torch.tensor([labels.index(row[0]) for row in rows])

    # 80%는 학습용, 20%는 시험용
    torch.manual_seed(0)
    perm = torch.randperm(len(y))
    n_test = max(1, len(y) // 5)
    test, train = perm[:n_test], perm[n_test:]

    model = build_model(len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        noisy = X[train] + torch.randn_like(X[train]) * 0.01  # 살짝 흔들어서 과적합 방지
        loss = loss_fn(model(noisy), y[train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(X[test]).argmax(1) == y[test]).float().mean().item()
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}  시험 정확도 {acc:.1%}")

    torch.save({"state_dict": model.state_dict(), "labels": labels}, MODEL_PATH)
    print(f"모델 저장: {MODEL_PATH}  제스처: {labels}")


if __name__ == "__main__":
    main()
