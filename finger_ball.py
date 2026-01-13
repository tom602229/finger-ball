"""
食指踢球應用 (v2 - 更好玩版)
- 用食指碰撞球，球會被踢走
- 球碰到牆壁會反彈
- [新] 左側設有球門，射門可以得分
- [新] 得分時會有慶祝特效和音效
- [新] 每次得分後，球會以隨機顏色重生
"""

import cv2
import mediapipe as mp
import time
import os
import math
import random
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === 安裝音效模組提示 ===
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("="*50)
    print("提示：如果想啟用音效，請安裝 pygame 模組。")
    print("安裝指令: pip install pygame")
    print("="*50)
    time.sleep(3)


# === 設定參數 ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'gesture_recognizer.task')
CONFIDENCE_THRESHOLD = 0.5

# 視窗尺寸
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

# 球的參數
BALL_RADIUS = 25
BALL_COLOR = (0, 100, 255)
FRICTION = 0.98
KICK_POWER = 15
BOUNCE_DAMPING = 0.8
HOLD_SPEED_THRESHOLD = 8
HOLD_DAMPING = 0.3
GRAVITY = 0.5

# 遊戲元素
GOAL_WIDTH = 15
GOAL_HEIGHT = 150
GOAL_Y = (FRAME_HEIGHT - GOAL_HEIGHT) // 2
GOAL_COLOR = (255, 255, 224)

# 特效參數
PARTICLE_LIFESPAN = 60  # 粒子持續幀數
PARTICLE_COUNT = 40

# === 初始化變數 ===
recognition_result = None

# 球的狀態
ball_x, ball_y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
ball_vx, ball_vy = 0, 0

# 食指位置追蹤
prev_finger_x, prev_finger_y = 0, 0

# 食指指尖的 landmark 索引
INDEX_FINGER_TIP = 8

# 手部連線定義
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
    (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
]

# 遊戲狀態
score = 0
goal_text_timer = 0
particles = []

# === 初始化音效 (如果可用) ===
if PYGAME_AVAILABLE:
    pygame.mixer.init()
    # 請將音效檔案放在與腳本相同的目錄下
    kick_sound = pygame.mixer.Sound("kick.wav") if os.path.exists("kick.wav") else None
    goal_sound = pygame.mixer.Sound("goal.wav") if os.path.exists("goal.wav") else None
    bounce_sound = pygame.mixer.Sound("bounce.wav") if os.path.exists("bounce.wav") else None
else:
    kick_sound, goal_sound, bounce_sound = None, None, None

def play_sound(sound):
    if sound:
        sound.play()

def save_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global recognition_result
    recognition_result = result

def get_random_color():
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def reset_ball():
    global ball_x, ball_y, ball_vx, ball_vy, BALL_COLOR
    ball_x, ball_y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    ball_vx, ball_vy = 0, 0
    BALL_COLOR = get_random_color()

def create_goal_particles(x, y):
    global particles
    particles = []
    for _ in range(PARTICLE_COUNT):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 8)
        particles.append({
            'x': x, 'y': y,
            'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
            'life': PARTICLE_LIFESPAN,
            'color': get_random_color()
        })

def update_and_draw_particles(frame):
    global particles
    active_particles = []
    for p in particles:
        p['x'] += p['vx']
        p['y'] += p['vy']
        p['vy'] += GRAVITY * 0.1 # 粒子受輕微重力影響
        p['life'] -= 1
        if p['life'] > 0:
            alpha = p['life'] / PARTICLE_LIFESPAN
            radius = int(BALL_RADIUS * 0.2 * alpha)
            cv2.circle(frame, (int(p['x']), int(p['y'])), radius, p['color'], -1)
            active_particles.append(p)
    particles = active_particles

def draw_landmarks_on_frame(frame, hand_landmarks_list):
    h, w, _ = frame.shape
    for hand_landmarks in hand_landmarks_list:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        for pt in points:
            cv2.circle(frame, pt, 5, (255, 0, 0), -1)
        cv2.circle(frame, points[INDEX_FINGER_TIP], 12, (0, 255, 255), -1)

def get_index_finger_position(hand_landmarks, frame_width, frame_height):
    index_tip = hand_landmarks[INDEX_FINGER_TIP]
    return int(index_tip.x * frame_width), int(index_tip.y * frame_height)

def check_collision(finger_x, finger_y, ball_x, ball_y, radius):
    dist = math.hypot(finger_x - ball_x, finger_y - ball_y)
    return dist < radius + 15

def main():
    global recognition_result, ball_x, ball_y, ball_vx, ball_vy, BALL_COLOR
    global prev_finger_x, prev_finger_y, score, goal_text_timer

    # 初始化 MediaPipe
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options, running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1, min_hand_detection_confidence=CONFIDENCE_THRESHOLD,
        min_hand_presence_confidence=CONFIDENCE_THRESHOLD,
        min_tracking_confidence=CONFIDENCE_THRESHOLD, result_callback=save_result)

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        print("="*50 + "\n食指踢球 v2 - 更好玩版!\n" + "="*50)
        print("將球踢進左方球門來得分！\n按 ESC 離開\n" + "="*50)

        reset_ball() # 初始化球

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

            # === 物理與遊戲邏輯 ===
            ball_vy += GRAVITY
            ball_x += ball_vx
            ball_y += ball_vy
            ball_vx *= FRICTION

            # 牆壁反彈
            if ball_x + BALL_RADIUS > w:
                ball_x = w - BALL_RADIUS; ball_vx *= -BOUNCE_DAMPING; play_sound(bounce_sound)
            if ball_y - BALL_RADIUS < 0:
                ball_y = BALL_RADIUS; ball_vy *= -BOUNCE_DAMPING; play_sound(bounce_sound)
            elif ball_y + BALL_RADIUS > h:
                ball_y = h - BALL_RADIUS; ball_vy *= -BOUNCE_DAMPING; play_sound(bounce_sound)

            # 球門檢查 (球的左邊界小於球門寬度，且y在球門範圍內)
            if ball_x - BALL_RADIUS < GOAL_WIDTH and GOAL_Y < ball_y < GOAL_Y + GOAL_HEIGHT:
                score += 1
                goal_text_timer = 60 # 顯示 "GOAL!" 60 幀
                play_sound(goal_sound)
                create_goal_particles(ball_x, ball_y)
                reset_ball()

            # 手指互動
            if recognition_result and recognition_result.hand_landmarks:
                draw_landmarks_on_frame(frame, recognition_result.hand_landmarks)
                hand_landmarks = recognition_result.hand_landmarks[0]
                finger_x, finger_y = get_index_finger_position(hand_landmarks, w, h)
                
                finger_vx = finger_x - prev_finger_x
                finger_vy = finger_y - prev_finger_y

                if check_collision(finger_x, finger_y, ball_x, ball_y, BALL_RADIUS):
                    finger_speed = math.hypot(finger_vx, finger_vy)
                    if finger_speed < HOLD_SPEED_THRESHOLD:
                        target_x, target_y = finger_x, finger_y - BALL_RADIUS - 10
                        ball_x += (target_x - ball_x) * HOLD_DAMPING
                        ball_y += (target_y - ball_y) * HOLD_DAMPING
                        ball_vx *= 0.5; ball_vy *= 0.5
                    else:
                        dx, dy = ball_x - finger_x, ball_y - finger_y
                        dist = math.hypot(dx, dy)
                        if dist > 0:
                            dx /= dist; dy /= dist
                            kick = max(KICK_POWER, finger_speed * 0.8)
                            ball_vx = dx * kick + finger_vx * 0.5
                            ball_vy = dy * kick + finger_vy * 0.5
                            ball_x = finger_x + dx * (BALL_RADIUS + 20)
                            ball_y = finger_y + dy * (BALL_RADIUS + 20)
                            play_sound(kick_sound)

                prev_finger_x, prev_finger_y = finger_x, finger_y

            # === 繪製畫面 ===
            # 繪製球門
            cv2.rectangle(frame, (0, GOAL_Y), (GOAL_WIDTH, GOAL_Y + GOAL_HEIGHT), GOAL_COLOR, -1)
            cv2.rectangle(frame, (0, GOAL_Y), (GOAL_WIDTH, GOAL_Y + GOAL_HEIGHT), (255,255,255), 2)


            # 繪製球
            cv2.circle(frame, (int(ball_x), int(ball_y)), BALL_RADIUS, BALL_COLOR, -1)
            cv2.circle(frame, (int(ball_x), int(ball_y)), BALL_RADIUS, (0, 0, 0), 2)

            # 繪製分數
            cv2.putText(frame, f"SCORE: {score}", (w - 150, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

            # 繪製 "GOAL!" 特效
            if goal_text_timer > 0:
                text_size = cv2.getTextSize("GOAL!", cv2.FONT_HERSHEY_TRIPLEX, 3, 5)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (h + text_size[1]) // 2
                alpha = 0.5 + (goal_text_timer / 120) # 漸隱效果
                color = (0, 255, int(255 * alpha))
                cv2.putText(frame, "GOAL!", (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 3, color, 5)
                goal_text_timer -= 1

            # 繪製粒子
            update_and_draw_particles(frame)

            cv2.imshow('Finger Ball v2 - Kick it!', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()