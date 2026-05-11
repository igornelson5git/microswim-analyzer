from PySide6.QtCore import Qt, QThread, Signal, Slot
import numpy as np
from AnalysisConfig import AnalysisConfig
from typing import Dict, List, Optional, Tuple
import cv2

class TrackerWorker(QThread):
    frame_ready = Signal(np.ndarray)
    metrics_ready = Signal(dict)
    finished_ok = Signal(str, str, dict)
    failed = Signal(str)
    status_changed = Signal(str)

    def __init__(self, config: AnalysisConfig, selected_summary_metrics: List[str]):
        super().__init__()
        self.config = config
        self.selected_summary_metrics = selected_summary_metrics
        self.stop_requested = False
        self.pause_requested = False
        self.reselect_requested = False

        self.records: List[Dict] = []
        self.manual_reselect_frames: List[int] = []
        self.current_frame: Optional[np.ndarray] = None
        self.frame_index = 0

    def request_stop(self):
        self.stop_requested = True

    def set_paused(self, paused: bool):
        self.pause_requested = paused

    def request_reselect(self):
        self.reselect_requested = True

    def create_tracker(self):
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        raise RuntimeError(
            "CSRT tracker not available. Install opencv-contrib-python, not plain opencv-python."
        )

    @staticmethod
    def is_valid_box(box, prev_box) -> bool:
        x, y, w, h = map(float, box)
        px, py, pw, ph = map(float, prev_box)

        area = w * h
        prev_area = pw * ph
        cx, cy = x + w / 2, y + h / 2
        pcx, pcy = px + pw / 2, py + ph / 2

        jump = np.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
        area_ratio = area / prev_area if prev_area > 0 else 1
        aspect = w / h if h > 0 else 999

        if jump > 80:
            return False
        if area_ratio < 0.4 or area_ratio > 2.5:
            return False
        if aspect < 0.2 or aspect > 5:
            return False
        return True

    def select_roi_blocking(self, frame: np.ndarray, title: str):
        display = cv2.resize(
            frame,
            None,
            fx=self.config.display_scale,
            fy=self.config.display_scale,
        )
        roi = cv2.selectROI(title, display, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(title)
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            raise RuntimeError("ROI selection cancelled or invalid.")
        return (
            int(x / self.config.display_scale),
            int(y / self.config.display_scale),
            int(w / self.config.display_scale),
            int(h / self.config.display_scale),
        )

    def run(self):
        cap = None
        try:
            cfg = self.config
            cap = cv2.VideoCapture(cfg.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                raise RuntimeError("Could not determine video FPS.")

            ret, first_frame = cap.read()
            if not ret:
                raise RuntimeError("Could not read video.")

            self.status_changed.emit("Select the paramecium ROI in the OpenCV window.")
            x, y, w, h = self.select_roi_blocking(first_frame, "Select paramecium")

            tracker = self.create_tracker()
            tracker.init(first_frame, (x, y, w, h))
            last_good_box = (x, y, w, h)

            metrics_engine = MetricsEngine(cfg, fps)
            prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            camera_x = 0.0
            camera_y = 0.0
            self.frame_index = 0

            while not self.stop_requested:
                if self.pause_requested:
                    self.msleep(50)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                self.current_frame = frame.copy()
                self.frame_index += 1
                current_time_s = self.frame_index / fps
                if current_time_s > cfg.analysis_duration_s:
                    break

                if self.reselect_requested:
                    self.status_changed.emit("Reselect the paramecium ROI in the OpenCV window.")
                    x, y, w, h = self.select_roi_blocking(frame, "Reselect paramecium")
                    tracker = self.create_tracker()
                    tracker.init(frame, (x, y, w, h))
                    last_good_box = (x, y, w, h)
                    self.manual_reselect_frames.append(self.frame_index)
                    self.reselect_requested = False

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                success, box = tracker.update(frame)

                if success and not self.is_valid_box(box, last_good_box):
                    success = False

                if success:
                    x, y, w, h = map(int, box)
                    cx = x + w / 2
                    cy = y + h / 2

                    mask = np.ones_like(prev_gray, dtype=np.uint8) * 255
                    margin = 40
                    mx1 = max(0, x - margin)
                    my1 = max(0, y - margin)
                    mx2 = min(mask.shape[1], x + w + margin)
                    my2 = min(mask.shape[0], y + h + margin)
                    mask[my1:my2, mx1:mx2] = 0

                    prev_pts = cv2.goodFeaturesToTrack(
                        prev_gray,
                        maxCorners=300,
                        qualityLevel=0.01,
                        minDistance=10,
                        mask=mask,
                    )

                    dx = 0.0
                    dy = 0.0
                    if prev_pts is not None:
                        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                            prev_gray,
                            gray,
                            prev_pts,
                            None,
                        )
                        if curr_pts is not None and status is not None:
                            good_prev = prev_pts[status.flatten() == 1]
                            good_curr = curr_pts[status.flatten() == 1]
                            if len(good_prev) >= 10:
                                shifts = good_curr - good_prev
                                dx = np.median(shifts[:, 0, 0])
                                dy = np.median(shifts[:, 0, 1])

                    camera_x += dx
                    camera_y += dy
                    corrected_x = cx - camera_x
                    corrected_y = cy - camera_y

                    self.records.append(
                        {
                            "frame": self.frame_index,
                            "time_s": current_time_s,
                            "raw_x_px": cx,
                            "raw_y_px": cy,
                            "camera_x_px": camera_x,
                            "camera_y_px": camera_y,
                            "corrected_x_px": corrected_x,
                            "corrected_y_px": corrected_y,
                            "tracking_success": True,
                        }
                    )

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        f"Camera x={camera_x:.2f}, y={camera_y:.2f}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    last_good_box = box
                else:
                    self.records.append(
                        {
                            "frame": self.frame_index,
                            "time_s": current_time_s,
                            "raw_x_px": np.nan,
                            "raw_y_px": np.nan,
                            "camera_x_px": camera_x,
                            "camera_y_px": camera_y,
                            "corrected_x_px": np.nan,
                            "corrected_y_px": np.nan,
                            "tracking_success": False,
                        }
                    )
                    cv2.putText(
                        frame,
                        "Tracking lost - click Reselect ROI",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

                prev_gray = gray.copy()

                live = metrics_engine.live_summary(self.records, self.manual_reselect_frames)
                self.metrics_ready.emit(live)
                self.frame_ready.emit(frame)

                delay_ms = max(1, int(1000 / fps))
                self.msleep(delay_ms)

            if not self.records:
                raise RuntimeError("No tracking records were produced.")

            df, summary_df = metrics_engine.final_summary(
                self.records,
                self.manual_reselect_frames,
            )

            if self.selected_summary_metrics:
                keep = [m for m in self.selected_summary_metrics if m in summary_df.columns]
                summary_df = summary_df[keep]

            base = os.path.splitext(os.path.basename(cfg.video_path))[0]
            output_dir = cfg.output_dir or os.path.dirname(cfg.video_path) or "."
            os.makedirs(output_dir, exist_ok=True)

            tracking_csv = os.path.join(output_dir, f"{base}_msa_tracking.csv")
            summary_csv = os.path.join(output_dir, f"{base}_msa_summary.csv")

            df.to_csv(tracking_csv, index=False)
            summary_df.to_csv(summary_csv, index=False)

            self.finished_ok.emit(tracking_csv, summary_csv, summary_df.iloc[0].to_dict())

        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()

