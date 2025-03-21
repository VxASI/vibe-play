Let’s update the README for your `vibe-controls` package to include the new `flappy` mode, along with its controls, while keeping the existing instructions for the `fly` and `shoot` modes. The README will now cover all three modes (`fly`, `shoot`, and `flappy`), providing clear usage instructions and control details for each.

Here’s the updated README:

---

# Vibe Controls

A Python package to control games using hand gestures via webcam. Supports three modes: `fly` for controlling a Three.js flying game (like fly.pieter.com), `shoot` for controlling first-person shooter (FPS) games, and `flappy` for controlling games like Flappy Bird.

## Installation

### Prerequisites
- **Python 3.11+**

### Step 1: Install UV

- **macOS:**
  ```bash
  brew install uv
  ```
- **Windows/Linux:**
  ```bash
  pip install uv
  ```

### Step 2: Run Vibe Controls

To run the app, use the `uvx` command with the `--mode` argument to specify the control mode (`fly`, `shoot`, or `flappy`). The app will activate your webcam for hand gesture detection.

#### Command Syntax
```bash
uvx vibe-controls --mode [fly|shoot|flappy]
```

- **Fly Mode** (default): Control a Three.js flying game.
  ```bash
  uvx vibe-controls --mode fly
  ```
- **Shoot Mode**: Control an FPS game.
  ```bash
  uvx vibe-controls --mode shoot
  ```
- **Flappy Mode**: Control a game like Flappy Bird.
  ```bash
  uvx vibe-controls --mode flappy
  ```

Wait until the webcam is turned on and a window titled "Hand Tracking - Fly Mode", "Hand Tracking - Shoot Mode", or "Hand Tracking - Flappy Mode" appears.

## Controls

### Fly Mode (`--mode fly`)
Designed for controlling a Three.js flying game like fly.pieter.com.

- **Hand Detection**:
  - **One Hand**: Holds the 'W' key to move forward.
  - **Two Hands**: Holds the Space key to perform an action (e.g., jump or boost, depending on the game).
  - **No Hands**: Releases all keys.

- **Gestures**:
  - **Flat Hand**: No directional input (neutral).
  - **Tilt Up**: Presses the Down arrow key (pitch down).
  - **Tilt Down**: Presses the Up arrow key (pitch up).
  - **Tilt Left**: Presses the 'D' key (yaw right).
  - **Tilt Right**: Presses the 'A' key (yaw left).
  - **Rotate Left**: Presses the Left arrow key (roll left).
  - **Rotate Right**: Presses the Right arrow key (roll right).

- **Exit**: Press the ESC key to close the app.

### Shoot Mode (`--mode shoot`)
Designed for controlling FPS games.

- **Hand Detection**:
  - **One Hand**: Holds the 'W' key to move forward.
  - **No Hands**: Releases the 'W' key.

- **Gestures**:
  - **Two Fingers Extended (Pointing)**: Moves the mouse in the direction you point.
    - Point up: Mouse moves up (look up).
    - Point down: Mouse moves down (look down).
    - Point left: Mouse moves left (look left).
    - Point right: Mouse moves right (look right).
    - Point diagonally: Mouse moves diagonally (look in that direction).
  - **Four Fingers Extended**: Holds the left mouse button to shoot.

- **Exit**: Press the ESC key to close the app.

### Flappy Mode (`--mode flappy`)
Designed for controlling games like Flappy Bird.

- **Gestures**:
  - **All Fingers Extended (Hand Spread)**: Triggers a flap by pressing and releasing the Space key. The hand should resemble a bird’s wing with fingers spread apart. A flap occurs approximately every 0.5 seconds while the gesture is held.

- **Exit**: Press the ESC key to close the app.

## Notes
- Ensure your webcam is properly connected and has sufficient lighting for accurate hand detection.
- For FPS games in `shoot` mode, you may need to adjust the in-game mouse sensitivity to match the gesture controls for a smoother experience.
- In `flappy` mode, the flapping frequency can be adjusted by modifying the cooldown period in the code if needed (default is 0.5 seconds between flaps).
- If gesture detection is inconsistent, ensure your hand is clearly visible to the webcam and matches the expected gesture for the mode.

---

### Explanation of Changes
- **Added Flappy Mode to Command Syntax**: Included `flappy` as a valid option for the `--mode` argument with an example command (`uvx vibe-controls --mode flappy`).
- **Added Flappy Mode Controls Section**: Created a new subsection under "Controls" for `flappy` mode, describing the gesture (all fingers extended and spread) and the action (Space key press for a flap every 0.5 seconds).
- **Updated Notes**: Added a note about the flapping frequency in `flappy` mode and the possibility to adjust the cooldown period in the code.
- **Kept Existing Sections**: Retained the instructions and controls for `fly` and `shoot` modes, ensuring the README covers all three modes comprehensively.
