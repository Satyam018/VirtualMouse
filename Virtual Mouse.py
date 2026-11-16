import argparse

import cv2
import mediapipe as mp
import pyautogui as pg
import Util as util
from pynput import keyboard
from pynput.mouse import Button, Controller

mpHands = mp.solutions.hands
mouse = Controller()

sensitivity = 1.0
SENSITIVITY_MIN = 0.1
SENSITIVITY_MAX = 3.0
SENSITIVITY_STEP = 0.1
quit=False
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)


def set_sensitivity(value):
    global sensitivity
    sensitivity = round(max(SENSITIVITY_MIN, min(SENSITIVITY_MAX, value)), 2)
    print(f"Sensitivity: {sensitivity:.2f}x  (+/- to adjust, 0 to reset)")


def handle_sensitivity_key(char):
    if char in ("+", "="):
        set_sensitivity(sensitivity + SENSITIVITY_STEP)
    elif char in ("-", "_"):
        set_sensitivity(sensitivity - SENSITIVITY_STEP)
    elif char == "0":
        set_sensitivity(1.0)


def on_key_press(key):
    try:
        char = key.char
    except AttributeError:
        return
    if char:
        handle_sensitivity_key(char)


def main():
    parser = argparse.ArgumentParser(description="Control the mouse with hand gestures.")
    parser.add_argument(
        "-s", "--sensitivity", type=float, default=1.0,
        help=f"initial cursor sensitivity ({SENSITIVITY_MIN}-{SENSITIVITY_MAX}, default 1.0)"
    )
    args = parser.parse_args()
    set_sensitivity(args.sensitivity)

    capture = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils

    if not capture.isOpened():
        print("Error: Camera not found or not accessible!")
        return

    listener = None
    try:
        listener = keyboard.Listener(on_press=on_key_press)
        listener.start()
    except Exception as exc:
        print(f"Warning: global sensitivity hotkeys unavailable ({exc}); use --sensitivity.")

    try:
        while True:
            ret, frame = capture.read()
            if not ret and quit() == False:
                break

            frame = cv2.flip(frame, 1)  # Flip for better user experience
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            processed = hands.process(frameRGB)
            land_marks_lists = []

            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    land_marks_lists.append((lm.x, lm.y))

                detect_gestures(frame, land_marks_lists, processed)

            # cv2.imshow('Video Frame', frame)

            # Proper exit condition
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key != 255:
                handle_sensitivity_key(chr(key))
    finally:
        if listener is not None:
            listener.stop()
        capture.release()
        cv2.destroyAllWindows()


def detect_gestures(frame, land_mark_list, processed):
    if len(land_mark_list) < 21:
        return

    index_tip = _finger_tip = find_finger_tip(processed)
    thumbcross = util.get_distance(land_mark_list[4], land_mark_list[5])
    index_angle = util.get_angle(land_mark_list[5], land_mark_list[6], land_mark_list[8])
    middle_angle = util.get_angle(land_mark_list[9], land_mark_list[10], land_mark_list[12])


    if thumbcross < 100 and index_angle > 130 and middle_angle > 130:
        move_mouse(index_tip)
    elif is_left_click(land_mark_list):
        mouse.press(Button.left)
        mouse.release(Button.left)
    elif is_left_right(land_mark_list):
        mouse.press(Button.right)
        mouse.release(Button.right)
    elif double_click(land_mark_list):
        pg.doubleClick()

def quit():
    user_input = input("Do you really want to quit? (yes/no): ").strip().lower()
    if user_input == 'yes':
        return True
    elif user_input == 'no':
        return False

def is_left_click(land_mark_list):
    index_angle = util.get_angle(land_mark_list[5], land_mark_list[6], land_mark_list[8])
    middle_angle = util.get_angle(land_mark_list[9], land_mark_list[10], land_mark_list[12])
    thumbcross = util.get_distance(land_mark_list[4], land_mark_list[5])
    return index_angle < 130 and middle_angle > 130 and thumbcross > 100


def is_left_right(land_mark_list):
    index_angle = util.get_angle(land_mark_list[5], land_mark_list[6], land_mark_list[8])
    middle_angle = util.get_angle(land_mark_list[9], land_mark_list[10], land_mark_list[12])
    thumbcross = util.get_distance(land_mark_list[4], land_mark_list[5])
    return index_angle > 130 and middle_angle < 130 and thumbcross > 100

def double_click(land_mark_list):
    index_angle = util.get_angle(land_mark_list[5], land_mark_list[6], land_mark_list[8])
    middle_angle = util.get_angle(land_mark_list[9], land_mark_list[10], land_mark_list[12])
    thumbcross = util.get_distance(land_mark_list[4], land_mark_list[5])
    return index_angle < 130 and middle_angle < 130 and thumbcross > 100



def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    else:
        return None


def move_mouse(index_tip):
    if index_tip is None:
        return

    screen_width, screen_height = pg.size()
    x = int(screen_width / 2 + (index_tip.x - 0.5) * screen_width * sensitivity)
    y = int(screen_height / 2 + (index_tip.y - 0.5) * screen_height * sensitivity)
    x = max(0, min(screen_width - 1, x))
    y = max(0, min(screen_height - 1, y))
    pg.moveTo(x, y)


if __name__ == '__main__':
    main()
