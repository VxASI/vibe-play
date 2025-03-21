# src/vibe_controls/fly_control.py
import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
import math
import logging
import sys
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fly_control.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize controllers
keyboard = KeyboardController()
mouse = MouseController()

# State variables
w_pressed = False
space_pressed = False

# --- Fly Mode Logic ---
def get_gesture_fly(landmarks):
    """Detect hand gestures for fly mode."""
    wrist = landmarks[0]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    thumb_mcp = landmarks[2]
    pinky_mcp = landmarks[17]
    
    fingertips_y = (index_tip.y + middle_tip.y + ring_tip.y + pinky_tip.y) / 4
    delta_y = fingertips_y - wrist.y
    fingertips_x = (index_tip.x + middle_tip.x + ring_tip.x + pinky_tip.x) / 4
    delta_x = fingertips_x - wrist.x
    roll_vector = (pinky_mcp.x - thumb_mcp.x, pinky_mcp.y - thumb_mcp.y)
    roll_angle = math.atan2(roll_vector[1], roll_vector[0]) * 180 / math.pi
    
    pitch_up_threshold = -0.1
    pitch_down_threshold = 0.1
    yaw_left_threshold = -0.1
    yaw_right_threshold = 0.1
    roll_left_threshold = -30
    roll_right_threshold = 30
    
    logger.debug(f"Delta Y: {delta_y:.3f}, Delta X: {delta_x:.3f}, Roll Angle: {roll_angle:.1f}°")
    
    if delta_y < pitch_up_threshold:
        return "pitch_up"
    elif delta_y > pitch_down_threshold:
        return "pitch_down"
    elif delta_x < yaw_left_threshold:
        return "yaw_left"
    elif delta_x > yaw_right_threshold:
        return "yaw_right"
    elif roll_angle < roll_left_threshold:
        return "roll_left"
    elif roll_angle > roll_right_threshold:
        return "roll_right"
    else:
        return "flat"

def run_fly_mode():
    """Run the fly mode controls."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam. Exiting.")
        sys.exit(1)

    current_key = None
    logger.info("Fly mode activated. Open fly.pieter.com and position your hand.")
    logger.info("Controls: Flat = W, Tilt up = Down, Tilt down = Up, Left = D, Right = A, Rotate left = Left, Rotate right = Right")
    logger.info("One hand = W held, Two hands = Space held. Press ESC to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame. Exiting.")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                gesture = get_gesture_fly(landmarks)
                logger.info(f"Detected gesture: {gesture}")
                
                key_map = {
                    "flat": None,
                    "pitch_up": Key.down,
                    "pitch_down": Key.up,
                    "yaw_left": 'd',
                    "yaw_right": 'a',
                    "roll_left": Key.left,
                    "roll_right": Key.right
                }
                
                new_key = key_map.get(gesture, None)
                if new_key != current_key:
                    if current_key:
                        keyboard.release(current_key)
                        logger.info(f"Released: {current_key}")
                    if new_key:
                        keyboard.press(new_key)
                        logger.info(f"Pressed: {new_key}")
                    current_key = new_key
                
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                if current_key:
                    keyboard.release(current_key)
                    logger.info(f"No hand detected. Released: {current_key}")
                    current_key = None
            
            # Handle W and Space keys
            global w_pressed, space_pressed
            if num_hands >= 1 and not w_pressed:
                keyboard.press('w')
                logger.info("One hand: Holding W")
                w_pressed = True
            elif num_hands < 1 and w_pressed:
                keyboard.release('w')
                logger.info("No hands: Released W")
                w_pressed = False
            
            if num_hands >= 2 and not space_pressed:
                keyboard.press(Key.space)
                logger.info("Two hands: Holding Space")
                space_pressed = True
            elif num_hands < 2 and space_pressed:
                keyboard.release(Key.space)
                logger.info("Less than 2 hands: Released Space")
                space_pressed = False
            
            cv2.imshow('Hand Tracking - Fly Mode', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                logger.info("ESC pressed. Exiting.")
                break

    finally:
        if current_key:
            keyboard.release(current_key)
        if w_pressed:
            keyboard.release('w')
        if space_pressed:
            keyboard.release(Key.space)
        cap.release()
        cv2.destroyAllWindows()
        logger.info("App closed.")

# --- Shoot Mode Logic ---
def run_shoot_mode():
    """Run the shoot mode controls using mouse for aiming and clicking."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam. Exiting.")
        sys.exit(1)

    logger.info("Shoot mode activated. Use hand to aim, fist to shoot. Press ESC to exit.")
    last_click_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame. Exiting.")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                wrist_x = landmarks[0].x  # 0 to 1
                wrist_y = landmarks[0].y  # 0 to 1
                
                # Map hand position to screen (assuming 1920x1080, adjust as needed)
                screen_width, screen_height = 1920, 1080
                mouse_x = wrist_x * screen_width
                mouse_y = wrist_y * screen_height
                mouse.position = (mouse_x, mouse_y)
                logger.debug(f"Mouse moved to ({mouse_x:.0f}, {mouse_y:.0f})")
                
                # Simple fist detection: check if fingers are folded
                index_tip_y = landmarks[8].y
                index_mcp_y = landmarks[5].y
                if index_tip_y > index_mcp_y:  # Finger folded down
                    current_time = cv2.getTickCount() / cv2.getTickFrequency()
                    if current_time - last_click_time > 0.5:  # Debounce clicks
                        mouse.click(Button.left, 1)
                        logger.info("Fist detected: Mouse clicked")
                        last_click_time = current_time
                
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            cv2.imshow('Hand Tracking - Shoot Mode', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                logger.info("ESC pressed. Exiting.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("App closed.")

# --- Main Entry Point ---
def main():
    """Parse command-line arguments and run the selected mode."""
    parser = argparse.ArgumentParser(description="Vibe Controls: Hand gesture controls for games.")
    parser.add_argument("--mode", choices=["fly", "shoot"], default="fly", help="Control mode (default: fly)")
    args = parser.parse_args()

    if args.mode == "fly":
        run_fly_mode()
    elif args.mode == "shoot":
        run_shoot_mode()

if __name__ == "__main__":
    main()