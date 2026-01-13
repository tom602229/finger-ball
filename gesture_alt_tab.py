"""
手勢控制 Alt+Tab 視窗切換應用
- Victory (剪刀手): 開啟/關閉 Alt+Tab 模式
- Thumbs_Up: 切換到右邊視窗 (方向鍵右)
- Thumbs_Down: 切換到左邊視窗 (方向鍵左)
"""

import cv2
import mediapipe as mp
import time
import os
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 取得腳本所在目錄
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'gesture_recognizer.task')

# 禁用 pyautogui 的安全暫停
pyautogui.PAUSE = 0.05
pyautogui.FAILSAFE = False

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# 全域狀態
recognition_result = None
alt_tab_mode = False  # 是否在 Alt+Tab 模式中
last_gesture = None
last_gesture_time = 0
GESTURE_COOLDOWN = 0.8  # 手勢冷卻時間(秒)，避免重複觸發


def save_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global recognition_result
    recognition_result = result


def draw_landmarks_on_frame(frame, hand_landmarks_list):
    h, w, _ = frame.shape
    for hand_landmarks in hand_landmarks_list:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        for pt in points:
            cv2.circle(frame, pt, 5, (255, 0, 0), -1)


def handle_gesture(gesture_name, score):
    """處理手勢動作"""
    global alt_tab_mode, last_gesture, last_gesture_time

    current_time = time.time()

    # Closed_Fist 重置狀態，允許下一次 Thumbs Up/Down
    if gesture_name == "Closed_Fist":
        if last_gesture in ["Thumb_Up", "Thumb_Down"]:
            last_gesture = None
            return "狀態重置 (準備下一次切換)"
        return None

    # 檢查冷卻時間
    if gesture_name == last_gesture and (current_time - last_gesture_time) < GESTURE_COOLDOWN:
        return None

    action = None

    if gesture_name == "Victory":
        if not alt_tab_mode:
            # 開啟 Alt+Tab 模式
            alt_tab_mode = True
            pyautogui.keyDown('alt')
            pyautogui.press('tab')
            action = "Alt+Tab 開啟"
        else:
            # 關閉 Alt+Tab 模式
            alt_tab_mode = False
            pyautogui.keyUp('alt')
            action = "Alt+Tab 確認"
        last_gesture = gesture_name
        last_gesture_time = current_time

    elif alt_tab_mode:
        # 只有在 Alt+Tab 模式中才處理左右切換
        if gesture_name == "Thumb_Up":
            pyautogui.press('right')
            action = "切換右 →"
            last_gesture = gesture_name
            last_gesture_time = current_time
        elif gesture_name == "Thumb_Down":
            pyautogui.press('left')
            action = "切換左 ←"
            last_gesture = gesture_name
            last_gesture_time = current_time

    return action


def main():
    global recognition_result, alt_tab_mode

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        result_callback=save_result
    )

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        last_action = ""
        action_display_time = 0

        print("=" * 50)
        print("手勢控制 Alt+Tab 視窗切換")
        print("=" * 50)
        print("Victory (剪刀手): 開啟/關閉 Alt+Tab")
        print("Thumbs Up: 切換到右邊視窗")
        print("Thumbs Down: 切換到左邊視窗")
        print("Closed Fist (握拳): 重置狀態，可再次切換")
        print("ESC: 退出程式")
        print("=" * 50)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp = time.time_ns() // 1_000_000
            recognizer.recognize_async(mp_image, timestamp)

            # 繪製狀態列
            status_color = (0, 255, 255) if alt_tab_mode else (128, 128, 128)
            status_text = "Alt+Tab MODE: ON" if alt_tab_mode else "Alt+Tab MODE: OFF"
            cv2.rectangle(frame, (0, 0), (640, 40), status_color, -1)
            cv2.putText(frame, status_text, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            if recognition_result:
                draw_landmarks_on_frame(frame, recognition_result.hand_landmarks)

                if recognition_result.gestures:
                    gesture = recognition_result.gestures[0][0]
                    gesture_name = gesture.category_name
                    score = gesture.score

                    # 顯示辨識到的手勢
                    cv2.putText(frame, f"{gesture_name} ({score:.2f})", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 處理手勢
                    if score > 0.7:
                        action = handle_gesture(gesture_name, score)
                        if action:
                            last_action = action
                            action_display_time = time.time()

            # 顯示最近執行的動作
            if time.time() - action_display_time < 2.0:
                cv2.putText(frame, f"Action: {last_action}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 顯示操作提示
            y_pos = 450
            cv2.putText(frame, "Victory: Toggle | Up/Down: Switch | Fist: Reset", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Gesture Alt+Tab Controller', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                # 確保退出時釋放 Alt 鍵
                if alt_tab_mode:
                    pyautogui.keyUp('alt')
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
