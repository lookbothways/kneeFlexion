import cv2
import numpy as np
import mediapipe as mp # 0.10.21 recommended
from collections import deque
import time
import datetime
import os
import csv

from mediapipe.python.solutions import pose as mp_pose

# --- CONFIGURATION ---
BUFFER_SIZE = 8
CSV_FILENAME = 'knee_session_log.csv'
TARGET_ANGLE = 90
HOLD_DURATION = 10
REST_DURATION = 5
MAX_REPS = 10

# Landmark Indices
HIP_IDX = 24
KNEE_IDX = 26
ANKLE_IDX = 28

# States
STATE_IDLE = "WAITING"
STATE_HOLDING = "HOLDING"
STATE_RESTING = "RESTING"
STATE_FINISHED = "FINISHED"

# Styling
TEXT_COLOR = (255, 255, 255)
UI_BG_COLOR = (128, 128, 128)
FONT = cv2.FONT_HERSHEY_DUPLEX  # Closest built-in to Arial


# ---------------------

def calculate_flexion(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return int(abs(180 - angle))


def get_average_point(buffer):
    if not buffer: return None
    avg_x = sum(p[0] for p in buffer) / len(buffer)
    avg_y = sum(p[1] for p in buffer) / len(buffer)
    return (int(avg_x), int(avg_y))


def log_final_session(max_angle):
    file_exists = os.path.isfile(CSV_FILENAME)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILENAME, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Max_Flexion_Reached'])
        writer.writerow([timestamp, f"{max_angle} deg"])


# Initialization
hip_buffer = deque(maxlen=BUFFER_SIZE)
knee_buffer = deque(maxlen=BUFFER_SIZE)
ankle_buffer = deque(maxlen=BUFFER_SIZE)
angle_buffer = deque(maxlen=BUFFER_SIZE)

current_state = STATE_IDLE
reps_completed = 0
timer_start = 0
session_max_flexion = 0
session_logged = False

cap = cv2.VideoCapture(0)

# Check if camera opened to get dimensions
ret, frame = cap.read()
if ret:
    h, w, _ = frame.shape
else:
    h, w = 480, 640  # Fallback

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)

        # 1. Create Double-Wide Canvas
        # Width is w*2, 3 channels for BGR
        canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
        canvas[:, w:] = UI_BG_COLOR  # Fill right side with 50% grey
        canvas[:, :w] = frame  # Put video on left side

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        smooth_angle = 0
        curr_time = time.time()
        ui_x_offset = w + 40  # Start text 40 pixels into the grey area

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Landmark Smoothing
            raw_hip = (lm[HIP_IDX].x * w, lm[HIP_IDX].y * h)
            raw_knee = (lm[KNEE_IDX].x * w, lm[KNEE_IDX].y * h)
            raw_ankle = (lm[ANKLE_IDX].x * w, lm[ANKLE_IDX].y * h)

            hip_buffer.append(raw_hip)
            knee_buffer.append(raw_knee)
            ankle_buffer.append(raw_ankle)

            s_hip = get_average_point(hip_buffer)
            s_knee = get_average_point(knee_buffer)
            s_ankle = get_average_point(ankle_buffer)

            # Angle Calculation
            current_angle = calculate_flexion(
                (s_hip[0] / w, s_hip[1] / h),
                (s_knee[0] / w, s_knee[1] / h),
                (s_ankle[0] / w, s_ankle[1] / h)
            )
            angle_buffer.append(current_angle)
            smooth_angle = int(sum(angle_buffer) / len(angle_buffer))

            # Update Session Max Flexion
            if smooth_angle > session_max_flexion:
                session_max_flexion = smooth_angle

            # Drawing Overlay on Video side (Left)
            cv2.line(canvas, s_hip, s_knee, (255, 255, 255), 3)
            cv2.line(canvas, s_knee, s_ankle, (255, 255, 255), 3)
            cv2.circle(canvas, s_knee, 8, (0, 255, 0), -1)

            # --- STATE MACHINE LOGIC ---
            if current_state == STATE_IDLE:
                if smooth_angle >= TARGET_ANGLE:
                    current_state = STATE_HOLDING
                    timer_start = curr_time

            elif current_state == STATE_HOLDING:
                if smooth_angle < TARGET_ANGLE:
                    current_state = STATE_IDLE
                else:
                    elapsed = curr_time - timer_start
                    remaining = max(0, HOLD_DURATION - int(elapsed))

                    # Large Countdown on UI side
                    one_third = h // 3
                    offset = h // 2 + one_third
                    cv2.putText(canvas, f"HOLD FOR: {remaining}s", (ui_x_offset, offset),
                                FONT, 1.5, TEXT_COLOR, 3, cv2.LINE_AA)

                    if elapsed >= HOLD_DURATION:
                        reps_completed += 1
                        if reps_completed >= MAX_REPS:
                            current_state = STATE_FINISHED
                        else:
                            current_state = STATE_RESTING
                            timer_start = curr_time

            elif current_state == STATE_RESTING:
                elapsed = curr_time - timer_start
                remaining = max(0, REST_DURATION - int(elapsed))
                one_third = h // 3
                offset = h // 2 + one_third
                cv2.putText(canvas, f"REST: {remaining}s", (ui_x_offset, offset),
                            FONT, 1.5, TEXT_COLOR, 3, cv2.LINE_AA)

                if elapsed >= REST_DURATION:
                    current_state = STATE_IDLE

        # --- UI TEXT (Right Side) ---
        cv2.putText(canvas, f"STATUS: {current_state}", (ui_x_offset, 60),
                    FONT, 1, TEXT_COLOR, 2, cv2.LINE_AA)

        cv2.putText(canvas, f"CURRENT FLEXION: {smooth_angle} deg", (ui_x_offset, 120),
                    FONT, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)

        cv2.putText(canvas, f"SESSION MAX: {session_max_flexion} deg", (ui_x_offset, 170),
                    FONT, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)

        cv2.putText(canvas, f"REPS: {reps_completed} / {MAX_REPS}", (ui_x_offset, 220),
                    FONT, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)

        # Session End & Logging
        if current_state == STATE_FINISHED:
            cv2.putText(canvas, "SESSION COMPLETE", (ui_x_offset, h // 2 + 80),
                        FONT, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            if not session_logged:
                log_final_session(session_max_flexion)
                session_logged = True

        cv2.imshow('Knee Flexion Trainer', canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()