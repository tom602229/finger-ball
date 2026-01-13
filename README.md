# finger-ball

用手指在鏡頭前「踢球」的小遊戲，透過 MediaPipe 手勢/手部 landmark 追蹤食指位置，讓球在畫面中被推/踢動，並把球射進左側球門得分。

## Demo

![Finger Ball Demo](assets/finger-ball.gif)

## 檔案說明

- **`finger_ball.py`**
  - 手指踢球遊戲（含球門得分、粒子特效、可選音效）
- **`gesture_alt_tab.py`**
  - 手勢控制 Alt+Tab 的小工具（Victory 開關、Thumb Up/Down 左右切換）
- **`gesture_recognizer.task`**
  - MediaPipe Gesture Recognizer 模型檔

## 需求

- Python 3.9+（建議）
- Webcam

### Python 套件

- `opencv-python`
- `mediapipe`
- `pyautogui`（只有 `gesture_alt_tab.py` 需要）
- `pygame`（可選，用於 `finger_ball.py` 音效）

## 安裝

在專案資料夾內執行：

```bash
pip install opencv-python mediapipe
```

如果要使用 Alt+Tab 手勢控制：

```bash
pip install pyautogui
```

如果要啟用音效（可選）：

```bash
pip install pygame
```

## 執行

### 手指踢球

```bash
python finger_ball.py
```

### 手勢控制 Alt+Tab

```bash
python gesture_alt_tab.py
```

## 操作方式

### finger_ball

- **用食指碰撞球**：球會被踢走（手指速度越快踢得越遠）
- **慢速貼近球**：球會有「黏住/帶球」的效果
- **球進左側球門**：得分並重生新顏色的球
- **ESC**：離開

### gesture_alt_tab

- **Victory（剪刀手）**：開啟/關閉 Alt+Tab 模式
- **Thumb_Up**：往右切換
- **Thumb_Down**：往左切換
- **Closed_Fist（握拳）**：重置狀態（允許下一次切換）
- **ESC**：離開