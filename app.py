# uvx: mediapipe opencv-python pynput
import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import math
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fly_control.log'),  # Log to a file
        logging.StreamHandler(sys.stdout)        # Log to console
    ]
)
logger = logging.getLogger(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize keyboard controller
keyboard = Controller()

# Function to detect hand gesture based on landmarks
def get_gesture(landmarks):
    wrist = landmarks[0]  # Wrist landmark
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    thumb_mcp = landmarks[2]
    pinky_mcp = landmarks[17]
    
    # Pitch: Compare fingertip y-coordinates to wrist (y increases downward)
    fingertips_y = (index_tip.y + middle_tip.y + ring_tip.y + pinky_tip.y) / 4
    delta_y = fingertips_y - wrist.y  # <0 if tilted up, >0 if tilted down
    
    # Yaw: Compare fingertip x-coordinates to wrist
    fingertips_x = (index_tip.x + middle_tip.x + ring_tip.x + pinky_tip.x) / 4
    delta_x = fingertips_x - wrist.x  # <0 if turned left, >0 if turned right
    
    # Roll: Angle of thumb-to-pinky MCP vector
    roll_vector = (pinky_mcp.x - thumb_mcp.x, pinky_mcp.y - thumb_mcp.y)
    roll_angle = math.atan2(roll_vector[1], roll_vector[0]) * 180 / math.pi  # Degrees
    
    # Thresholds (adjust these based on testing)
    pitch_up_threshold = -0.1    # Hand tilted upward
    pitch_down_threshold = 0.1   # Hand tilted downward
    yaw_left_threshold = -0.1    # Hand turned left
    yaw_right_threshold = 0.1    # Hand turned right
    roll_left_threshold = -30    # Hand rotated left
    roll_right_threshold = 30    # Hand rotated right
    
    # Log the raw values for debugging
    logger.debug(f"Delta Y: {delta_y:.3f}, Delta X: {delta_x:.3f}, Roll Angle: {roll_angle:.1f}°")
    
    # Determine gesture
    if delta_y < pitch_up_threshold:
        return "pitch_up"        # Down arrow
    elif delta_y > pitch_down_threshold:
        return "pitch_down"      # Up arrow
    elif delta_x < yaw_left_threshold:
        return "yaw_left"        # A key
    elif delta_x > yaw_right_threshold:
        return "yaw_right"       # D key
    elif roll_angle < roll_left_threshold:
        return "roll_left"       # Left arrow
    elif roll_angle > roll_right_threshold:
        return "roll_right"      # Right arrow
    else:
        return "flat"            # W key

# Main loop
cap = cv2.VideoCapture(0)  # Open webcam
if not cap.isOpened():
    logger.error("Failed to open webcam. Exiting.")
    sys.exit(1)

current_key = None  # Track currently pressed key

logger.info("Fly mode activated. Open fly.pieter.com and position your hand.")
logger.info("Flat hand = W, Tilt up = Down, Tilt down = Up, Left = A, Right = D, Rotate left = Left arrow, Rotate right = Right arrow")
logger.info("Press ESC to exit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame. Exiting.")
            break
        
        # Process frame with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Use first hand detected
            landmarks = hand_landmarks.landmark
            gesture = get_gesture(landmarks)
            logger.info(f"Detected gesture: {gesture}")
            
            # Map gestures to keys
            key_map = {
                "flat": 'w',
                "pitch_up": Key.down,
                "pitch_down": Key.up,
                "yaw_left": 'd',
                "yaw_right": 'a',
                "roll_left": Key.left,
                "roll_right": Key.right
            }
            
            new_key = key_map.get(gesture, None)
            
            # Update key presses
            if new_key != current_key:
                if current_key is not None:
                    logger.info(f"Releasing key: {current_key}")
                    keyboard.release(current_key)
                if new_key is not None:
                    logger.info(f"Pressing key: {new_key}")
                    keyboard.press(new_key)
                current_key = new_key
            
            # Draw landmarks for debugging
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            # No hand detected, release any pressed key
            if current_key is not None:
                logger.info(f"No hand detected. Releasing key: {current_key}")
                keyboard.release(current_key)
                current_key = None
        
        # Display the frame
        cv2.imshow('Hand Tracking - Fly Mode', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
            logger.info("ESC pressed. Exiting.")
            break

finally:
    # Cleanup
    if current_key is not None:
        logger.info(f"Releasing final key: {current_key}")
        keyboard.release(current_key)
    cap.release()
    cv2.destroyAllWindows()
    logger.info("App closed.")