from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import cv2

from camera_manager import CameraConfig, CameraManager
from gesture_detector import Gesture, GestureConfig, GestureDetector, GestureState
from hand_tracker import HandTracker, HandTrackerConfig
from mouse_controller import ControlBox, MouseController, MouseControllerConfig


WINDOW_NAME = "macOS Gesture Mouse"


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    tracker: HandTrackerConfig
    gestures: GestureConfig
    mouse: MouseControllerConfig
    control_margin_x: int = 120
    control_margin_y: int = 90


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Touchless macOS mouse control with OpenCV, MediaPipe Hands, and PyAutoGUI."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--target-fps", type=int, default=60)
    parser.add_argument("--inference-width", type=int, default=640)
    parser.add_argument("--inference-height", type=int, default=480)
    parser.add_argument("--smoothing", type=float, default=7.5)
    parser.add_argument("--drag-smoothing", type=float, default=4.0)
    parser.add_argument("--fast-smoothing", type=float, default=2.6)
    parser.add_argument("--click-threshold", type=float, default=35.0)
    parser.add_argument("--pinch-release-threshold", type=float, default=50.0)
    parser.add_argument("--right-click-threshold", type=float, default=40.0)
    parser.add_argument("--drag-hold", type=float, default=0.70)
    parser.add_argument("--scroll-sensitivity", type=float, default=0.22)
    parser.add_argument("--control-margin-x", type=int, default=120)
    parser.add_argument("--control-margin-y", type=int, default=90)
    parser.add_argument("--disable-failsafe", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        camera=CameraConfig(
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            target_fps=args.target_fps,
            inference_width=args.inference_width,
            inference_height=args.inference_height,
        ),
        tracker=HandTrackerConfig(),
        gestures=GestureConfig(
            click_threshold=args.click_threshold,
            pinch_release_threshold=args.pinch_release_threshold,
            right_click_threshold=args.right_click_threshold,
            drag_hold_seconds=args.drag_hold,
            scroll_scale=args.scroll_sensitivity,
        ),
        mouse=MouseControllerConfig(
            smoothing_factor=args.smoothing,
            drag_smoothing_factor=args.drag_smoothing,
            fast_smoothing_factor=args.fast_smoothing,
            fail_safe=not args.disable_failsafe,
        ),
        control_margin_x=args.control_margin_x,
        control_margin_y=args.control_margin_y,
    )


def make_control_box(frame_shape: Tuple[int, int, int], margin_x: int, margin_y: int) -> ControlBox:
    height, width = frame_shape[:2]
    margin_x = min(max(0, margin_x), max(0, width // 2 - 2))
    margin_y = min(max(0, margin_y), max(0, height // 2 - 2))
    return ControlBox(
        left=margin_x,
        top=margin_y,
        right=width - margin_x,
        bottom=height - margin_y,
    )


def draw_overlay(
    frame,
    control_box: ControlBox,
    state: GestureState,
    fps: float,
    cursor_position: Tuple[int, int],
    paused: bool,
    smoothing_factor: float,
) -> None:
    color = (60, 220, 120)
    if paused:
        color = (80, 180, 255)
    if state.gesture in {Gesture.LEFT_CLICK, Gesture.DOUBLE_CLICK, Gesture.RIGHT_CLICK, Gesture.DRAG}:
        color = (30, 190, 255)
    elif state.gesture == Gesture.NONE:
        color = (120, 120, 120)

    cv2.rectangle(frame, (control_box.left, control_box.top), (control_box.right, control_box.bottom), color, 2)

    if state.index_tip is not None:
        cv2.circle(frame, state.index_tip, 8, (255, 255, 255), -1)
        cv2.circle(frame, state.index_tip, 10, color, 2)

    lines = [
        f"Status: {'Paused' if paused else 'Active'}",
        f"Gesture: {state.gesture.value}",
        f"FPS: {fps:.1f}",
        f"Smooth: {smoothing_factor:.1f}",
        f"Pinch I: {state.thumb_index_distance:.1f}px",
        f"Pinch M: {state.thumb_middle_distance:.1f}px",
        f"Cursor: {cursor_position[0]}, {cursor_position[1]}",
        "Q/Esc quit  P pause  R reset  +/- smooth",
    ]

    for index, text in enumerate(lines):
        y = 28 + index * 24
        draw_text(frame, text, (16, y), scale=0.62)

    draw_gesture_guide(frame)
    if paused:
        draw_pause_banner(frame)


def draw_text(frame, text: str, origin: Tuple[int, int], scale: float = 0.52) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)


def draw_gesture_guide(frame) -> None:
    guide_lines = [
        "Gesture Guide",
        "Move: index finger up",
        "Click: quick thumb + index pinch",
        "Open: two quick thumb + index pinches",
        "Drag: hold thumb + index pinch, then move",
        "Right click: thumb + middle pinch",
        "Scroll: index + middle up, move smoothly",
    ]
    height, width = frame.shape[:2]
    panel_left = 12
    panel_top = max(150, height - 174)
    panel_right = min(width - 12, 624)
    panel_bottom = height - 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_left, panel_top), (panel_right, panel_bottom), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (panel_left, panel_top), (panel_right, panel_bottom), (80, 220, 140), 1)

    for index, text in enumerate(guide_lines):
        y = panel_top + 24 + index * 20
        scale = 0.54 if index == 0 else 0.46
        draw_text(frame, text, (panel_left + 12, y), scale=scale)


def draw_pause_banner(frame) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height // 2 - 28), (width, height // 2 + 28), (0, 90, 170), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    draw_text(frame, "PAUSED - press P to resume", (24, height // 2 + 8), scale=0.72)


def print_macos_permission_notice() -> None:
    print(
        "\nmacOS setup needed for mouse control:\n"
        "1. Open System Settings.\n"
        "2. Go to Privacy & Security > Accessibility.\n"
        "3. Enable Terminal, your IDE, or the Python launcher running this app.\n"
        "4. Go to Privacy & Security > Camera and allow the same app if prompted.\n"
    )


def apply_gesture_actions(
    state: GestureState,
    mouse: MouseController,
    control_box: ControlBox,
) -> Tuple[int, int]:
    cursor_position = (-1, -1)
    active = state.index_tip is not None

    if active and state.drag_started:
        mouse.mouse_down()

    if active:
        dragging = state.gesture == Gesture.DRAG or state.drag_active or state.drag_released
        if state.gesture in {Gesture.MOVE, Gesture.LEFT_CLICK, Gesture.DRAG} or dragging:
            cursor_position = mouse.move_from_camera_point(state.index_tip, control_box, dragging=dragging)
        elif state.gesture == Gesture.SCROLL:
            mouse.scroll(state.scroll_amount)

    if state.drag_released or (state.drag_active and not active):
        mouse.mouse_up()

    if state.double_click_ready:
        mouse.double_click()
    elif state.click_ready:
        mouse.click()

    if active and state.right_click_ready:
        mouse.right_click()

    if state.gesture == Gesture.NONE or not active:
        mouse.reset_smoothing()

    return cursor_position


def run(config: AppConfig) -> None:
    print_macos_permission_notice()
    camera = CameraManager(config.camera)
    tracker = HandTracker(config.tracker)
    detector = GestureDetector(config.gestures)
    mouse = MouseController(config.mouse)

    try:
        camera.open()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        paused = False

        while True:
            ok, frame, inference_frame = camera.read()
            if not ok or frame is None or inference_frame is None:
                print("Camera frame was not available; retrying.")
                continue

            control_box = make_control_box(inference_frame.shape, config.control_margin_x, config.control_margin_y)
            hand, results = tracker.process(inference_frame)

            if paused:
                state = detector.detect(None)
                cursor_position = (-1, -1)
            else:
                state = detector.detect(hand)
                cursor_position = apply_gesture_actions(state, mouse, control_box)

            preview = frame
            if frame.shape[:2] != inference_frame.shape[:2]:
                preview = cv2.resize(frame, (inference_frame.shape[1], inference_frame.shape[0]))

            tracker.draw_landmarks(preview, results)
            draw_overlay(preview, control_box, state, camera.fps, cursor_position, paused, mouse.config.smoothing_factor)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("p"), ord("P")):
                paused = not paused
                mouse.reset()
                detector.reset()
                print("Gesture control paused." if paused else "Gesture control resumed.")
            elif key in (ord("r"), ord("R")):
                mouse.reset()
                detector.reset()
                print("Gesture state reset.")
            elif key in (ord("+"), ord("=")):
                print(f"Smoothing: {mouse.adjust_smoothing(0.5):.1f}")
            elif key in (ord("-"), ord("_")):
                print(f"Smoothing: {mouse.adjust_smoothing(-0.5):.1f}")
    except KeyboardInterrupt:
        print("Gesture mouse stopped by keyboard interrupt.")
    except RuntimeError as exc:
        print(f"Startup error: {exc}")
    finally:
        mouse.release()
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = build_parser().parse_args()
    run(config_from_args(args))


if __name__ == "__main__":
    main()
