import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize webcam and FaceMesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Track click cooldowns
last_left_click_time = 0
last_right_click_time = 0
click_delay = 1  # seconds

while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)

    frame_h, frame_w, _ = frame.shape
    landmark_points = output.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Cursor movement - use landmark[475]
        eye_landmark = landmarks[475]
        x = int(eye_landmark.x * frame_w)
        y = int(eye_landmark.y * frame_h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        screen_x = screen_w * eye_landmark.x
        screen_y = screen_h * eye_landmark.y
        pyautogui.moveTo(screen_x, screen_y)

        # Left Eye Blink - landmarks 145 (top), 159 (bottom)
        left_top = landmarks[145]
        left_bottom = landmarks[159]
        left_eye_diff = abs(left_top.y - left_bottom.y)

        # Right Eye Blink - landmarks 374 (top), 386 (bottom)
        right_top = landmarks[374]
        right_bottom = landmarks[386]
        right_eye_diff = abs(right_top.y - right_bottom.y)

        # Iris Scroll Control - use landmark[473]
        iris = landmarks[473]
        iris_y = iris.y

        # Draw reference points
        for lm in [left_top, left_bottom, right_top, right_bottom, iris]:
            x = int(lm.x * frame_w)
            y = int(lm.y * frame_h)
            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)

        current_time = time.time()

        # Left Click
        if left_eye_diff < 0.004 and (current_time - last_left_click_time > click_delay):
            pyautogui.click(button='left')
            print("Left Click")
            last_left_click_time = current_time

        # Right Click
        if right_eye_diff < 0.004 and (current_time - last_right_click_time > click_delay):
            pyautogui.click(button='right')
            print("Right Click")
            last_right_click_time = current_time

        # Scroll (up/down)
        if iris_y < 0.40:
            pyautogui.scroll(20)  # Scroll up
        elif iris_y > 0.60:
            pyautogui.scroll(-20)  # Scroll down

    # Show frame
    cv2.imshow('Eye Controlled Mouse', frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()