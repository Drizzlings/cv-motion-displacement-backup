import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


Point = Tuple[float, float]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


@dataclass
class Detection:
    center: Point
    bbox: Tuple[int, int, int, int]
    points: Optional[np.ndarray] = None


@dataclass
class CalibrationResult:
    homography: np.ndarray
    corners: np.ndarray
    source_frame_index: int
    status: str


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": output_dir,
        "videos": output_dir / "videos",
        "plots": output_dir / "plots",
        "calibration": output_dir / "calibration",
        "data": output_dir / "data",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def transform_point(homography: np.ndarray, point: Point) -> Point:
    src = np.array([[[point[0], point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, homography)[0, 0]
    return float(dst[0]), float(dst[1])


def load_calibration_points(path: Path) -> Dict[str, List[List[float]]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("points", {})


def save_calibration_points(path: Path, points: Dict[str, List[List[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"points": points}, f, allow_unicode=True, sort_keys=False)


def load_target_rois(path: Path) -> Dict[str, List[int]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("rois", {})


def save_target_rois(path: Path, rois: Dict[str, List[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"rois": rois}, f, allow_unicode=True, sort_keys=False)


def read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_index} from {video_path.name}")
    return frame


def draw_click_preview(frame: np.ndarray, points: Sequence[Point], video_name: str) -> np.ndarray:
    preview = frame.copy()
    labels = ["TL", "TR", "BR", "BL"]
    for i, p in enumerate(points):
        pi = (int(round(p[0])), int(round(p[1])))
        cv2.circle(preview, pi, 6, (0, 255, 255), -1)
        cv2.putText(preview, labels[i], (pi[0] + 8, pi[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    if len(points) >= 2:
        for a, b in zip(points[:-1], points[1:]):
            cv2.line(preview, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 0, 255), 2, cv2.LINE_AA)
    if len(points) == 4:
        cv2.line(
            preview,
            (int(points[3][0]), int(points[3][1])),
            (int(points[0][0]), int(points[0][1])),
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    help_text = f"{video_name}: click TL, TR, BR, BL | r reset | q skip | enter save"
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(preview, help_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return preview


def mark_points_for_video(video_path: Path, frame_index: int) -> List[List[float]]:
    frame = read_video_frame(video_path, frame_index)
    window = f"mark calibration - {video_path.name}"
    clicked: List[Point] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(frame.shape[1], 1280), min(frame.shape[0], 800))
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, draw_click_preview(frame, clicked, video_path.name))
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and len(clicked) == 4:
            break
        if key == ord("r"):
            clicked.clear()
        if key == ord("q") or key == 27:
            cv2.destroyWindow(window)
            raise RuntimeError(f"calibration marking skipped for {video_path.name}")
    cv2.destroyWindow(window)
    return [[round(float(x), 3), round(float(y), 3)] for x, y in clicked]


def mark_calibration_points(config: dict, selected_video: Optional[str] = None) -> None:
    input_dir = project_path(config["input_dir"])
    points_file = project_path(config["calibration"]["points_file"])
    points = load_calibration_points(points_file)
    videos = sorted(input_dir.glob("*.mp4"))
    if selected_video:
        videos = [v for v in videos if v.name == selected_video or v.stem == selected_video]
        if not videos:
            raise FileNotFoundError(f"video not found: {selected_video}")
    frame_index = int(config["calibration"].get("mark_frame_index", 0))
    for video in videos:
        points[video.name] = mark_points_for_video(video, frame_index)
        save_calibration_points(points_file, points)
        print(f"[INFO] Saved 4 calibration points for {video.name} to {points_file}")


def ensure_calibration_points(config: dict, videos: Sequence[Path]) -> None:
    points_file = project_path(config["calibration"]["points_file"])
    points = load_calibration_points(points_file)
    missing = [video for video in videos if len(points.get(video.name, [])) != 4]
    if not missing:
        return

    print("[INFO] Some videos do not have 4 calibration points. Marking them now.")
    frame_index = int(config["calibration"].get("mark_frame_index", 0))
    for video in missing:
        points[video.name] = mark_points_for_video(video, frame_index)
        save_calibration_points(points_file, points)
        print(f"[INFO] Saved 4 calibration points for {video.name} to {points_file}")


def select_target_roi(video_path: Path, frame_index: int) -> List[int]:
    frame = read_video_frame(video_path, frame_index)
    window = f"select target ROI - {video_path.name}"
    roi = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise RuntimeError(f"target ROI was not selected for {video_path.name}")
    return [x, y, w, h]


def mark_target_rois(config: dict, selected_video: Optional[str] = None) -> None:
    input_dir = project_path(config["input_dir"])
    roi_file = project_path(config["optical_flow"]["target_rois_file"])
    rois = load_target_rois(roi_file)
    videos = sorted(input_dir.glob("*.mp4"))
    if selected_video:
        videos = [v for v in videos if v.name == selected_video or v.stem == selected_video]
        if not videos:
            raise FileNotFoundError(f"video not found: {selected_video}")
    frame_index = int(config["optical_flow"].get("init_frame_index", 0))
    for video in videos:
        rois[video.name] = select_target_roi(video, frame_index)
        save_target_rois(roi_file, rois)
        print(f"[INFO] Saved target ROI for {video.name} to {roi_file}")


def ensure_target_rois(config: dict, videos: Sequence[Path]) -> None:
    roi_file = project_path(config["optical_flow"]["target_rois_file"])
    rois = load_target_rois(roi_file)
    missing = [video for video in videos if len(rois.get(video.name, [])) != 4]
    if not missing:
        return

    print("[INFO] Some videos do not have a target ROI. Selecting them now.")
    frame_index = int(config["optical_flow"].get("init_frame_index", 0))
    for video in missing:
        rois[video.name] = select_target_roi(video, frame_index)
        save_target_rois(roi_file, rois)
        print(f"[INFO] Saved target ROI for {video.name} to {roi_file}")


def calibrate_video(video_path: Path, config: dict) -> CalibrationResult:
    points_file = project_path(config["calibration"]["points_file"])
    all_points = load_calibration_points(points_file)
    manual = all_points.get(video_path.name)
    if manual is None:
        raise RuntimeError(f"missing calibration points for {video_path.name}")
    if len(manual) != 4:
        raise RuntimeError(f"{points_file} must contain exactly 4 points for {video_path.name}")

    width_mm = float(config["calibration"]["reference_width_mm"])
    height_mm = float(config["calibration"]["reference_height_mm"])
    image_points = np.array(manual, dtype=np.float32).reshape(4, 2)
    world_points = np.array([[0, 0], [width_mm, 0], [width_mm, height_mm], [0, height_mm]], dtype=np.float32)
    homography, _ = cv2.findHomography(image_points, world_points)
    if homography is None:
        raise RuntimeError(f"failed to compute homography for {video_path.name}")
    return CalibrationResult(homography, image_points, int(config["calibration"].get("mark_frame_index", 0)), "manual")


def get_feature_params(config: dict) -> dict:
    flow = config["optical_flow"]
    return {
        "maxCorners": int(flow.get("max_corners", 80)),
        "qualityLevel": float(flow.get("quality_level", 0.01)),
        "minDistance": float(flow.get("min_distance", 5)),
        "blockSize": int(flow.get("block_size", 7)),
    }


def get_lk_params(config: dict) -> dict:
    flow = config["optical_flow"]
    win = flow.get("win_size", [21, 21])
    return {
        "winSize": (int(win[0]), int(win[1])),
        "maxLevel": int(flow.get("max_level", 3)),
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }


def clamp_roi(roi: Sequence[int], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = [int(v) for v in roi]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def detect_features_in_roi(gray: np.ndarray, roi: Tuple[int, int, int, int], config: dict) -> Optional[np.ndarray]:
    x, y, w, h = roi
    mask = np.zeros_like(gray)
    mask[y : y + h, x : x + w] = 255
    features = cv2.goodFeaturesToTrack(gray, mask=mask, **get_feature_params(config))
    if features is not None and len(features) >= 2:
        return features.astype(np.float32)

    # Fallback grid points still feed the LK tracker, but do not use color or thresholding.
    xs = np.linspace(x + w * 0.25, x + w * 0.75, 3)
    ys = np.linspace(y + h * 0.25, y + h * 0.75, 3)
    grid = np.array([[[px, py]] for py in ys for px in xs], dtype=np.float32)
    return grid


def track_target_with_optical_flow(video_path: Path, config: dict) -> Tuple[List[Optional[Detection]], List[Optional[Point]]]:
    roi_file = project_path(config["optical_flow"]["target_rois_file"])
    rois = load_target_rois(roi_file)
    if video_path.name not in rois:
        raise RuntimeError(f"missing target ROI for {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    init_frame_index = int(config["optical_flow"].get("init_frame_index", 0))
    init_roi = clamp_roi(rois[video_path.name], width, height)
    init_center = (init_roi[0] + init_roi[2] / 2.0, init_roi[1] + init_roi[3] / 2.0)
    min_tracks = int(config["optical_flow"].get("min_tracks", 6))
    fb_limit = float(config["optical_flow"].get("max_forward_backward_error", 2.0))
    max_center_jump = float(config["optical_flow"].get("max_center_jump_px", 80.0))

    detections: List[Optional[Detection]] = []
    centers: List[Optional[Point]] = []
    prev_gray: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None
    center: Optional[Point] = None

    for frame_index in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_index < init_frame_index:
            detections.append(None)
            centers.append(None)
            prev_gray = gray
            continue

        if frame_index == init_frame_index or points is None or center is None or len(points) < min_tracks:
            if center is None:
                center = init_center
            roi = (
                int(round(center[0] - init_roi[2] / 2.0)),
                int(round(center[1] - init_roi[3] / 2.0)),
                init_roi[2],
                init_roi[3],
            )
            points = detect_features_in_roi(gray, clamp_roi(roi, width, height), config)
            prev_gray = gray
            bbox = (
                int(round(center[0] - init_roi[2] / 2.0)),
                int(round(center[1] - init_roi[3] / 2.0)),
                init_roi[2],
                init_roi[3],
            )
            detections.append(Detection(center, bbox, points.reshape(-1, 2) if points is not None else None))
            centers.append(center)
            continue

        next_points, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None, **get_lk_params(config))
        if next_points is None or status is None:
            detections.append(None)
            centers.append(None)
            prev_gray = gray
            points = None
            continue

        back_points, back_status, _back_err = cv2.calcOpticalFlowPyrLK(gray, prev_gray, next_points, None, **get_lk_params(config))
        status_mask = status.reshape(-1).astype(bool)
        if back_points is not None and back_status is not None:
            fb_error = np.linalg.norm(points.reshape(-1, 2) - back_points.reshape(-1, 2), axis=1)
            status_mask &= back_status.reshape(-1).astype(bool) & (fb_error <= fb_limit)

        old_good = points.reshape(-1, 2)[status_mask]
        new_good = next_points.reshape(-1, 2)[status_mask]
        if len(new_good) < min_tracks:
            detections.append(None)
            centers.append(None)
            prev_gray = gray
            points = None
            continue

        delta = np.median(new_good - old_good, axis=0)
        if float(np.linalg.norm(delta)) > max_center_jump:
            detections.append(None)
            centers.append(None)
            prev_gray = gray
            points = None
            continue
        center = (float(center[0] + delta[0]), float(center[1] + delta[1]))
        if center[0] < -init_roi[2] or center[0] > width + init_roi[2] or center[1] < -init_roi[3] or center[1] > height + init_roi[3]:
            detections.append(None)
            centers.append(None)
            prev_gray = gray
            points = None
            continue
        points = new_good.reshape(-1, 1, 2).astype(np.float32)
        prev_gray = gray
        bbox = (
            int(round(center[0] - init_roi[2] / 2.0)),
            int(round(center[1] - init_roi[3] / 2.0)),
            init_roi[2],
            init_roi[3],
        )
        detections.append(Detection(center, bbox, new_good))
        centers.append(center)

    cap.release()
    return detections, centers


def moving_average(points: Sequence[Optional[Point]], window: int) -> List[Optional[Point]]:
    if window <= 1:
        return list(points)
    smoothed: List[Optional[Point]] = []
    valid: List[Point] = []
    for p in points:
        if p is not None:
            valid.append(p)
            valid = valid[-window:]
            xs = [v[0] for v in valid]
            ys = [v[1] for v in valid]
            smoothed.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        else:
            smoothed.append(None)
    return smoothed


def draw_calibration(video_path: Path, output_path: Path, calibration: CalibrationResult) -> None:
    frame = read_video_frame(video_path, calibration.source_frame_index)
    corners = calibration.corners.astype(np.int32)
    cv2.polylines(frame, [corners], True, (0, 0, 255), 3, cv2.LINE_AA)
    for i, p in enumerate(corners):
        cv2.circle(frame, tuple(p), 6, (0, 255, 255), -1)
        cv2.putText(frame, str(i + 1), tuple(p + np.array([6, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Calibration: manual points", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    ok, encoded = cv2.imencode(output_path.suffix, frame)
    if ok:
        encoded.tofile(str(output_path))


def transform_and_filter_points(
    homography: np.ndarray,
    centers: Sequence[Optional[Point]],
    config: dict,
) -> List[Optional[Point]]:
    max_step = float(config["quality"].get("max_world_step_mm", 200.0))
    max_abs = float(config["quality"].get("max_abs_world_coord_mm", 5000.0))
    world_points: List[Optional[Point]] = []
    previous: Optional[Point] = None
    for center in centers:
        if center is None:
            world_points.append(None)
            continue
        world = transform_point(homography, center)
        if abs(world[0]) > max_abs or abs(world[1]) > max_abs:
            world_points.append(None)
            continue
        if previous is not None and math.dist(world, previous) > max_step:
            world_points.append(None)
            continue
        world_points.append(world)
        previous = world
    return world_points


def write_plot(video_name: str, times: List[float], displacements: List[Optional[float]], output_path: Path) -> None:
    valid_times = [t for t, d in zip(times, displacements) if d is not None]
    valid_values = [d for d in displacements if d is not None]
    plt.figure(figsize=(8, 4.5))
    if valid_times:
        plt.plot(valid_times, valid_values, color="#1f77b4", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Straight displacement (mm)")
    plt.title(f"{video_name} displacement curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def process_video(video_path: Path, config: dict, dirs: Dict[str, Path]) -> dict:
    name = video_path.stem
    calibration = calibrate_video(video_path, config)
    draw_calibration(video_path, dirs["calibration"] / f"{name}_calibration.jpg", calibration)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(dirs["videos"] / f"{name}_annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    cap.release()

    detections, raw_centers = track_target_with_optical_flow(video_path, config)
    smooth_centers = moving_average(raw_centers, int(config["optical_flow"].get("smooth_window", 5)))
    world_points = transform_and_filter_points(calibration.homography, smooth_centers, config)

    first_world = next((p for p in world_points if p is not None), None)
    straight: List[Optional[float]] = []
    cumulative: List[Optional[float]] = []
    total_path = 0.0
    previous_world: Optional[Point] = None
    for p in world_points:
        if p is None or first_world is None:
            straight.append(None)
            cumulative.append(None)
            continue
        if previous_world is not None:
            total_path += math.dist(p, previous_world)
        previous_world = p
        straight.append(math.dist(p, first_world))
        cumulative.append(total_path)

    cap = cv2.VideoCapture(str(video_path))
    trajectory: List[Tuple[int, int]] = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det = detections[frame_index] if frame_index < len(detections) else None
        center = smooth_centers[frame_index] if frame_index < len(smooth_centers) else None
        if center is not None:
            pi = (int(round(center[0])), int(round(center[1])))
            trajectory.append(pi)
            cv2.circle(frame, pi, 5, (0, 255, 255), -1)
        if det is not None:
            x, y, w, h = det.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 80, 0), 2)
            if det.points is not None:
                for px, py in det.points.reshape(-1, 2):
                    cv2.circle(frame, (int(round(px)), int(round(py))), 2, (0, 255, 0), -1)
        for a, b in zip(trajectory[:-1], trajectory[1:]):
            cv2.line(frame, a, b, (0, 0, 255), 2, cv2.LINE_AA)
        if straight[frame_index] is not None:
            cv2.putText(frame, f"Disp: {straight[frame_index]:.1f} mm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, f"Path: {cumulative[frame_index]:.1f} mm", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        writer.write(frame)
        frame_index += 1
    cap.release()
    writer.release()

    times = [i / fps for i in range(len(straight))]
    write_plot(name, times, straight, dirs["plots"] / f"{name}_displacement.png")

    data_path = dirs["data"] / f"{name}_timeseries.csv"
    with data_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["frame", "time_s", "pixel_x", "pixel_y", "world_x_mm", "world_y_mm", "straight_mm", "cumulative_mm"])
        for i, (pc, wp, sd, cd) in enumerate(zip(smooth_centers, world_points, straight, cumulative)):
            writer_csv.writerow(
                [
                    i,
                    f"{times[i]:.6f}",
                    "" if pc is None else f"{pc[0]:.3f}",
                    "" if pc is None else f"{pc[1]:.3f}",
                    "" if wp is None else f"{wp[0]:.3f}",
                    "" if wp is None else f"{wp[1]:.3f}",
                    "" if sd is None else f"{sd:.3f}",
                    "" if cd is None else f"{cd:.3f}",
                ]
            )

    valid_count = sum(1 for p in world_points if p is not None)
    valid_ratio = valid_count / max(len(world_points), 1)
    valid_straight = [d for d in straight if d is not None]
    final_disp = valid_straight[-1] if valid_straight else 0.0
    max_disp = max(valid_straight) if valid_straight else 0.0
    status = "ok" if valid_ratio >= float(config["quality"]["min_valid_ratio"]) else "warning_low_valid_ratio"
    return {
        "video": video_path.name,
        "fps": round(fps, 3),
        "frames": frame_count,
        "valid_frames": valid_count,
        "valid_ratio": round(valid_ratio, 4),
        "calibration_status": calibration.status,
        "final_displacement_mm": round(final_disp, 3),
        "max_displacement_mm": round(max_disp, 3),
        "cumulative_path_mm": round(total_path, 3),
        "status": status,
    }


def write_results_csv(rows: List[dict], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def process_all(config: dict, selected_video: Optional[str] = None) -> None:
    input_dir = project_path(config["input_dir"])
    dirs = ensure_dirs(project_path(config["output_dir"]))
    videos = sorted(input_dir.glob("*.mp4"))
    if selected_video:
        videos = [v for v in videos if v.name == selected_video or v.stem == selected_video]
    if not videos:
        raise FileNotFoundError(f"No matching .mp4 files found in {input_dir}")

    ensure_calibration_points(config, videos)
    ensure_target_rois(config, videos)

    rows: List[dict] = []
    failures: List[Tuple[str, str]] = []
    for video in videos:
        try:
            print(f"[INFO] Processing {video.name}")
            rows.append(process_video(video, config, dirs))
        except Exception as exc:
            failures.append((video.name, str(exc)))
            rows.append(
                {
                    "video": video.name,
                    "fps": 0,
                    "frames": 0,
                    "valid_frames": 0,
                    "valid_ratio": 0,
                    "calibration_status": "failed",
                    "final_displacement_mm": 0,
                    "max_displacement_mm": 0,
                    "cumulative_path_mm": 0,
                    "status": f"failed: {exc}",
                }
            )
    write_results_csv(rows, dirs["root"] / "results.csv")
    print(f"[INFO] Measurement outputs written to {dirs['root']}")
    if failures:
        for video, reason in failures:
            print(f"[WARN] {video}: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure planar displacement of a moving object from videos.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML configuration file.")
    parser.add_argument("--video", help="Optional video filename or stem.")
    parser.add_argument("--remark", action="store_true", help="Re-click calibration corners and target ROI before measurement.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.remark:
        mark_calibration_points(config, args.video)
        mark_target_rois(config, args.video)
    process_all(config, args.video)


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
    main()
