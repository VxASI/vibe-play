# src/vibe_controls/fly_control.py
import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
import math
import logging
import sys
import argparse
import time  # For cooldown

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
mouse_button_pressed = False
last_avg_x = None
last_avg_y = None
HORIZONTAL_SENSITIVITY = 50  # Adjusted based on smaller horizontal movements
VERTICAL_SENSITIVITY = 100   # Adjusted based on larger vertical movements
ALPHA = 0.5  # Increased for more responsiveness (less smoothing)

# Utility function to count extended fingers
def count_extended_fingers(landmarks):
    fingers = [
        (8, 6),   # Index finger: tip to PIP
        (12, 10), # Middle finger
        (16, 14), # Ring finger
        (20, 18), # Pinky
    ]
    extended = 0
    for tip, pip in fingers:
        if landmarks[tip].y < landmarks[pip].y:  # Tip above PIP means extended
            extended += 1
    return extended

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
    """Run the shoot mode controls for FPS games."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam. Exiting.")
        sys.exit(1)

    global mouse_button_pressed, w_pressed, last_direction_x, last_direction_y
    mouse_button_pressed = False
    w_pressed = False
    last_direction_x = 0
    last_direction_y = 0
    
    logger.info("Shoot mode activated. Point with two fingers to aim, extend four fingers to shoot. Press ESC to exit.")

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
                extended = count_extended_fingers(landmarks)
                logger.debug(f"Extended fingers: {extended}")
                
                if extended == 2:
                    # Two fingers: Calculate pointing direction
                    index_tip = landmarks[8]   # Index fingertip
                    middle_tip = landmarks[12] # Middle fingertip
                    wrist = landmarks[0]       # Wrist
                    
                    # Average position of the two fingertips
                    avg_x = (index_tip.x + middle_tip.x) / 2
                    avg_y = (index_tip.y + middle_tip.y) / 2
                    
                    # Direction vector from wrist to average fingertip position
                    direction_x = avg_x - wrist.x
                    direction_y = avg_y - wrist.y
                    
                    # Normalize the direction vector to ensure consistent speed
                    magnitude = math.sqrt(direction_x**2 + direction_y**2)
                    if magnitude > 0:
                        direction_x /= magnitude
                        direction_y /= magnitude
                    else:
                        direction_x = 0
                        direction_y = 0
                    
                    # Scale direction with sensitivity
                    dx = direction_x * HORIZONTAL_SENSITIVITY
                    dy = direction_y * VERTICAL_SENSITIVITY
                    
                    # Apply smoothing to reduce jitter
                    smoothed_dx = ALPHA * dx + (1 - ALPHA) * last_direction_x
                    smoothed_dy = ALPHA * dy + (1 - ALPHA) * last_direction_y
                    
                    # Move mouse in the pointing direction
                    mouse.move(int(smoothed_dx), int(smoothed_dy))
                    logger.debug(f"Pointing direction: dx={smoothed_dx:.2f}, dy={smoothed_dy:.2f}")
                    
                    # Update last direction for smoothing
                    last_direction_x = smoothed_dx
                    last_direction_y = smoothed_dy
                
                elif extended == 4:
                    # Four fingers: Hold left mouse button to shoot
                    if not mouse_button_pressed:
                        mouse.press(Button.left)
                        mouse_button_pressed = True
                        logger.info("Four fingers: Holding mouse left button")
                else:
                    # Release mouse button if not four fingers
                    if mouse_button_pressed:
                        mouse.release(Button.left)
                        mouse_button_pressed = False
                        logger.info("Released mouse left button")
                
                # Draw hand landmarks for visual feedback
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                # No hands detected: Release controls and reset direction
                if mouse_button_pressed:
                    mouse.release(Button.left)
                    mouse_button_pressed = False
                    logger.info("No hands: Released mouse left button")
                last_direction_x = 0
                last_direction_y = 0
            
            # Handle 'W' key for forward movement
            if num_hands >= 1 and not w_pressed:
                keyboard.press('w')
                w_pressed = True
                logger.info("One hand: Holding W")
            elif num_hands < 1 and w_pressed:
                keyboard.release('w')
                w_pressed = False
                logger.info("No hands: Released W")
            
            cv2.imshow('Hand Tracking - Shoot Mode', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                logger.info("ESC pressed. Exiting.")
                break

    finally:
        # Cleanup: Release all controls
        if mouse_button_pressed:
            mouse.release(Button.left)
        if w_pressed:
            keyboard.release('w')
        cap.release()
        cv2.destroyAllWindows()
        logger.info("App closed.")

# --- Flappy Mode Logic ---
def run_flappy_mode():
    """Run the flappy mode controls for games like Flappy Bird."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam. Exiting.")
        sys.exit(1)

    global space_pressed
    space_pressed = False
    last_flap_time = 0  # Track the last time a flap was triggered
    FLAP_COOLDOWN = 0.5  # Cooldown period in seconds between flaps

    logger.info("Flappy mode activated. Hold a hand with all fingers extended to flap (press Space). Press ESC to exit.")

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
                extended = count_extended_fingers(landmarks)
                
                if extended == 4:  # Check for hand with all fingers extended (like a bird's wing)
                    # logger.info(f"Hand with {extended} fingers detected: flapping gesture recognized")
                    current_time = time.time()
                    if current_time - last_flap_time >= FLAP_COOLDOWN:
                        # Trigger a flap (press and release Space)
                        keyboard.press(Key.space)
                        keyboard.release(Key.space)
                        logger.info("Flap detected: Space pressed")
                        last_flap_time = current_time
                # else:
                #     # Log if the gesture doesn't match
                #     # logger.info(f"Hand with {extended} fingers detected: not a flapping gesture")
                
                # Draw hand landmarks for visual feedback
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # else:
            #     # No hands detected
            #     # logger.info("No hands detected")
            
            # cv2.imshow('Hand Tracking - Flappy Mode', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                logger.info("ESC pressed. Exiting.")
                break

    finally:
        # Cleanup: Release all controls
        cap.release()
        cv2.destroyAllWindows()
        logger.info("App closed.")

# --- Main Entry Point ---
def main():
    """Parse command-line arguments and run the selected mode."""
    parser = argparse.ArgumentParser(description="Vibe Controls: Hand gesture controls for games.")
    parser.add_argument("--mode", choices=["fly", "shoot", "flappy"], default="fly", help="Control mode (default: fly)")
    args = parser.parse_args()

    if args.mode == "fly":
        run_fly_mode()
    elif args.mode == "shoot":
        run_shoot_mode()
    elif args.mode == "flappy":
        run_flappy_mode()

if __name__ == "__main__":
    main()