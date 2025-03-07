import cv2
import mediapipe as mp
import pyautogui as pg
import Util as util
from pynput.mouse import Button, Controller

mpHands = mp.solutions.hands
mouse = Controller()
quit=False
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)


def main():
    capture = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils

    if not capture.isOpened():
        print("Error: Camera not found or not accessible!")
        return

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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def detect_gestures(frame, land_mark_list, processed):
    if len(land_mark_list) < 21:
        return

    index_tip = _finger_tip = find_finger_tip(processed)
    thumbcross = util.get_distance(land_mark_list[4], land_mark_list[5])
    index_angle = util.get_angle(land_mark_list[5], land_mark_list[6], land_mark_list[8])
    middle_angle = util.get_angle(land_mark_list[5], land_mark_list[6], land_mark_list[8])


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
    screen_width, screen_height = pg.size()

    if index_tip is not None:
        sensitivity=1
        x = int(index_tip.x * screen_width*sensitivity)
        y = int(index_tip.y * screen_height*sensitivity)
        pg.moveTo(x, y)


if __name__ == '__main__':
    main()
