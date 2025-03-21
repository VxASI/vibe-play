I understand you want to update the README to reflect the usage of the `--mode` argument and include the controls for both `fly` and `shoot` modes in your `vibe-controls` package. Since the code itself doesn’t need changes, I’ll focus on updating the README to provide clear instructions for running the app in either mode and explain the controls for each.

Here’s the updated README:

---

# Vibe Controls

A Python package to control games using hand gestures via webcam. Supports two modes: `fly` for controlling a Three.js flying game (like fly.pieter.com) and `shoot` for controlling first-person shooter (FPS) games.

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

To run the app, use the `uvx` command with the `--mode` argument to specify the control mode (`fly` or `shoot`). The app will activate your webcam for hand gesture detection.

#### Command Syntax
```bash
uvx vibe-controls --mode [fly|shoot]
```

- **Fly Mode** (default): Control a Three.js flying game.
  ```bash
  uvx vibe-controls --mode fly
  ```
- **Shoot Mode**: Control an FPS game.
  ```bash
  uvx vibe-controls --mode shoot
  ```

Wait until the webcam is turned on and a window titled "Hand Tracking - Fly Mode" or "Hand Tracking - Shoot Mode" appears.

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

## Notes
- Ensure your webcam is properly connected and has sufficient lighting for accurate hand detection.
- For FPS games, you may need to adjust the in-game mouse sensitivity to match the gesture controls for a smoother experience.
- If the mouse control in `shoot` mode doesn’t work as expected in-game, try testing outside the game first to confirm gesture detection.

---

### Explanation of Changes
- **Added Mode Selection Instructions**: Included the `--mode` argument in the "Run Vibe Controls" section with examples for both `fly` and `shoot` modes.
- **Detailed Controls Section**: Added a new "Controls" section with subsections for `fly` and `shoot` modes, explaining hand detection and gesture controls for each.
- **Additional Notes**: Provided tips for troubleshooting, such as ensuring proper lighting and adjusting in-game sensitivity.
