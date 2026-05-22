# macOS Gesture Mouse

High-performance hand gesture mouse control for macOS on Apple Silicon Macs. The app uses OpenCV for the FaceTime camera, MediaPipe Hands for real-time hand tracking, and PyAutoGUI for system cursor control.

## Install

```bash
python3.10 -m venv .venv310
source .venv310/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Equivalent package-by-package install:

```bash
pip install opencv-python
pip install mediapipe
pip install pyautogui
pip install numpy
```

Optional helper libraries:

```bash
pip install pynput
pip install imutils
```

## macOS Permissions

PyAutoGUI needs Accessibility permission before it can move or click the mouse.

1. Open `System Settings`.
2. Go to `Privacy & Security`.
3. Open `Accessibility`.
4. Enable the terminal, IDE, or Python launcher that runs this app.
5. Open `Camera` in the same section and allow camera access if macOS prompts for it.

## Run

```bash
cd "/Users/aditya/Developer/hand gesture "
deactivate 2>/dev/null || true
source .venv310/bin/activate
python main.py
```

Hotkeys:

- `q` or `Esc`: quit.
- `p`: pause or resume gesture control.
- `r`: reset drag/click state and cursor smoothing.
- `+` / `-`: increase or decrease cursor smoothing while the app is running.

Useful tuning flags:

```bash
python main.py --width 640 --height 480 --target-fps 60 --smoothing 7.5
python main.py --width 1280 --height 720 --inference-width 640 --inference-height 480
python main.py --click-threshold 35 --pinch-release-threshold 50 --drag-hold 0.70
python main.py --scroll-sensitivity 0.22 --drag-smoothing 4.0 --fast-smoothing 2.6
python main.py --control-margin-x 140 --control-margin-y 105
```

## Gestures

- `Move`: hold only the index finger up. The green control box maps to the whole screen, and moving past its edge pins the cursor to the screen edge.
- `Left click`: pinch thumb and index briefly, then release.
- `Double click / open`: do two quick thumb + index pinches.
- `Drag`: pinch thumb and index, hold until Drag appears, then move and release.
- `Right click`: pinch thumb and middle finger.
- `Scroll`: hold index and middle fingers up, then move vertically.

## Architecture

- `camera_manager.py`: OpenCV camera setup through `cv2.CAP_AVFOUNDATION`, frame flipping, resizing, and FPS measurement.
- `hand_tracker.py`: MediaPipe Hands initialization, landmark extraction, and skeleton drawing.
- `gesture_detector.py`: gesture state machine, pinch distances, finger extension, scroll deltas, and debouncing.
- `mouse_controller.py`: Retina-aware screen mapping through `pyautogui.size()`, cursor movement, clicks, scrolling, and drag state.
- `smoothing_filter.py`: configurable exponential moving average filter for jitter reduction.
- `main.py`: application loop, CLI configuration, overlay drawing, graceful shutdown, and macOS permission guidance.

## Performance Notes

- Default capture is `640x480` with a `640x480` inference frame for low latency.
- You can capture at `720p` while keeping inference at `640x480` for better preview quality without greatly increasing MediaPipe cost.
- `cv2.CAP_AVFOUNDATION` is used for the macOS camera backend.
- `pyautogui.size()` returns macOS logical display coordinates, which works correctly with Retina scaling for cursor movement.
- `pyautogui.FAILSAFE` is enabled by default. Move the cursor to a screen corner to stop PyAutoGUI actions if something goes wrong.
# hand-gesture-mouse
