# mediapipe-hand

[MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/guide) 기반 웹캠 실시간 **Hand Landmarker** / **Gesture Recognizer** 예제.

## 설치

```bash
pip install -U -r requirements.txt
```

테스트 환경: Windows 11, Python 3.14, mediapipe 1.1.0, OpenCV 5.0

## 모델

모델 파일은 저장소에 포함되어 있습니다 (스크립트와 같은 폴더에 있어야 함).

| 파일 | 크기 | 출처 |
| --- | --- | --- |
| `hand_landmarker.task` | 7.8 MB | [Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) — [다운로드](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) |
| `gesture_recognizer.task` | 8.4 MB | [Gesture Recognizer](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer) — [다운로드](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task) |

## 스크립트

| 파일 | 설명 |
| --- | --- |
| `hand_webcam.py` | 손 21개 랜드마크 + 연결선, 왼손/오른손(Left/Right)과 신뢰도, FPS·손 개수 표시 |
| `gesture_webcam.py` | 손 랜드마크 + 제스처 이름과 점수, 왼손/오른손 표시 |
| `hand_gesture_webcam.py` | 웹캠 1대로 두 모델을 동시에 실행. 왼쪽: Hand Landmarker, 오른쪽: Gesture Recognizer |

공통 설정: 최대 2손 인식, `RunningMode.VIDEO`, 거울 모드(좌우 반전), 검출/추적 신뢰도 0.5.

## 사용 예

```bash
python hand_webcam.py
python gesture_webcam.py
python hand_gesture_webcam.py
```

키: `q` / `ESC` 종료

## 인식 제스처 (Gesture Recognizer)

| 이름 | 제스처 |
| --- | --- |
| `Closed_Fist` | ✊ 주먹 |
| `Open_Palm` | ✋ 손바닥 펴기 |
| `Pointing_Up` | ☝️ 검지 위로 |
| `Thumb_Up` | 👍 엄지 위로 |
| `Thumb_Down` | 👎 엄지 아래로 |
| `Victory` | ✌️ 브이 |
| `ILoveYou` | 🤟 사랑해 |
| `None` | 인식되지 않음 |

## 손 랜드마크 번호

```
0  WRIST
1  THUMB_CMC          2  THUMB_MCP          3  THUMB_IP           4  THUMB_TIP
5  INDEX_FINGER_MCP   6  INDEX_FINGER_PIP   7  INDEX_FINGER_DIP   8  INDEX_FINGER_TIP
9  MIDDLE_FINGER_MCP  10 MIDDLE_FINGER_PIP  11 MIDDLE_FINGER_DIP  12 MIDDLE_FINGER_TIP
13 RING_FINGER_MCP    14 RING_FINGER_PIP    15 RING_FINGER_DIP    16 RING_FINGER_TIP
17 PINKY_MCP          18 PINKY_PIP          19 PINKY_DIP          20 PINKY_TIP
```

## 참고 / 문제 해결

- **한글 경로:** MediaPipe는 Windows에서 한글 등 비ASCII 문자가 들어간 경로의 모델 파일을 열지 못합니다 (`RuntimeError: Unable to open file`). 그래서 모든 스크립트는 `model_asset_path` 대신 파일을 바이트로 읽어 `model_asset_buffer`로 전달합니다.
- **웹캠 공유:** Windows에서는 한 프로그램만 웹캠을 쓸 수 있습니다. 두 모델을 같이 쓰려면 스크립트 2개를 동시에 띄우지 말고 `hand_gesture_webcam.py`를 실행하세요.
- **웹캠이 열리지 않을 때:** 스크립트 상단의 `CAMERA_INDEX`를 `1` 등으로 바꿔 보세요. 웹캠은 `cv2.CAP_DSHOW`(DirectShow)로 엽니다.
- **로그 경고:** 실행 시 나오는 `NORM_RECT without IMAGE_DIMENSIONS`, `Feedback manager requires ...` 메시지는 MediaPipe 내부 경고이며 동작에는 영향이 없습니다.
- `hand_gesture_webcam.py`는 프레임마다 두 모델을 실행하므로 단일 스크립트보다 FPS가 낮습니다.
