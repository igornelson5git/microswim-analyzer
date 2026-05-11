import sys
import os
from dataclasses import dataclass

from typing import Dict, List, Optional, Tuple
from AnalysisConfig import AnalysisConfig
from TrackerWorker import TrackerWorker
from MetricsEngine import MetricsEngine 
import cv2
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph as pg
except ImportError:
    pg = None


SUMMARY_METRICS = {
    "duration_s": "Duration (s)",
    "mean_speed_um_s": "Mean speed (um/s)",
    "median_speed_um_s": "Median speed (um/s)",
    "std_speed_um_s": "Speed SD (um/s)",
    "p95_speed_um_s": "P95 speed (um/s)",
    "total_path_um": "Total path (um)",
    "net_displacement_um": "Net displacement (um)",
    "persistence_length_proxy": "Persistence proxy (um/s)",
    "meandering_index": "Meandering index",
    "median_turn_deg_per_s": "Median turn (deg/s)",
    "mean_velocity_parallel_field_um_s": "Mean parallel field velocity (um/s)",
    "abs_velocity_parallel_field_um_s": "Abs parallel field velocity (um/s)",
    "mean_velocity_perpendicular_field_um_s": "Mean perpendicular field velocity (um/s)",
    "abs_velocity_perpendicular_field_um_s": "Abs perpendicular field velocity (um/s)",
    "mean_field_alignment": "Mean field alignment",
    "std_field_alignment": "Field alignment SD",
    "manual_reselect_events": "Manual reselections",
    "excluded_reselect_frames": "Excluded reselect frames",
    "valid_frames_used": "Valid frames used",
}


LIVE_METRICS = [
    "time_s",
    "speed_smooth_um_s",
    "mean_speed_um_s",
    "median_speed_um_s",
    "p95_speed_um_s",
    "total_path_um",
    "net_displacement_um",
    "meandering_index",
    "mean_abs_turn_deg",
    "mean_velocity_parallel_field_um_s",
    "mean_field_alignment",
    "excluded_frames",
]

class MicroSwim(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MicroSwim Analyzer")
        self.resize(1350, 850)
        self.worker: Optional[TrackerWorker] = None

        self.video_path_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()

        self.display_scale_spin = self.double_spin(0.1, 2.0, 0.5, 0.1)
        self.microns_spin = self.double_spin(0.0001, 10000.0, 1.0, 0.1)
        self.exclusion_spin = self.double_spin(0.0, 10.0, 0.5, 0.1)
        self.duration_spin = self.double_spin(1.0, 36000.0, 60.0, 1.0)
        self.field_angle_spin = self.double_spin(-360.0, 360.0, 0.0, 1.0)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 999)
        self.window_spin.setValue(7)

        self.start_btn = QPushButton("Start")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")
        self.reselect_btn = QPushButton("Reselect ROI")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.reselect_btn.setEnabled(False)

        self.video_label = QLabel("Choose a video, then click Start.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(760, 520)
        self.video_label.setStyleSheet("background: #111; color: #ddd; border: 1px solid #333;")

        self.status_label = QLabel("Idle")
        self.live_labels: Dict[str, QLabel] = {}
        self.metric_checks: Dict[str, QCheckBox] = {}

        self.speed_curve = None
        self.turn_curve = None
        self.speed_plot = None
        self.turn_plot = None
        self.plot_times: List[float] = []
        self.plot_speeds: List[float] = []
        self.plot_turns: List[float] = []

        self.build_ui()
        self.connect_signals()

    @staticmethod
    def double_spin(minimum, maximum, value, step):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(4)
        return spin

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_scroll.setWidget(left_widget)
        left_scroll.setFixedWidth(390)

        file_group = QGroupBox("Files")
        file_layout = QGridLayout(file_group)
        browse_video_btn = QPushButton("Browse video")
        browse_output_btn = QPushButton("Output folder")
        file_layout.addWidget(QLabel("Video"), 0, 0)
        file_layout.addWidget(self.video_path_edit, 0, 1)
        file_layout.addWidget(browse_video_btn, 1, 1)
        file_layout.addWidget(QLabel("Output"), 2, 0)
        file_layout.addWidget(self.output_dir_edit, 2, 1)
        file_layout.addWidget(browse_output_btn, 3, 1)
        self.browse_video_btn = browse_video_btn
        self.browse_output_btn = browse_output_btn

        params_group = QGroupBox("Analysis parameters")
        params_layout = QFormLayout(params_group)
        params_layout.addRow("Display scale", self.display_scale_spin)
        params_layout.addRow("Microns / pixel", self.microns_spin)
        params_layout.addRow("Reselect exclusion (s)", self.exclusion_spin)
        params_layout.addRow("Analysis duration (s)", self.duration_spin)
        params_layout.addRow("Field angle (deg)", self.field_angle_spin)
        params_layout.addRow("Smoothing window", self.window_spin)

        buttons_group = QGroupBox("Controls")
        buttons_layout = QGridLayout(buttons_group)
        buttons_layout.addWidget(self.start_btn, 0, 0)
        buttons_layout.addWidget(self.pause_btn, 0, 1)
        buttons_layout.addWidget(self.stop_btn, 1, 0)
        buttons_layout.addWidget(self.reselect_btn, 1, 1)

        metrics_group = QGroupBox("Summary metrics to export")
        metrics_layout = QVBoxLayout(metrics_group)
        select_all_btn = QPushButton("Select all")
        select_none_btn = QPushButton("Select none")
        metric_btn_row = QHBoxLayout()
        metric_btn_row.addWidget(select_all_btn)
        metric_btn_row.addWidget(select_none_btn)
        metrics_layout.addLayout(metric_btn_row)
        self.select_all_btn = select_all_btn
        self.select_none_btn = select_none_btn

        for key, label in SUMMARY_METRICS.items():
            cb = QCheckBox(label)
            cb.setChecked(True)
            self.metric_checks[key] = cb
            metrics_layout.addWidget(cb)

        left_layout.addWidget(file_group)
        left_layout.addWidget(params_group)
        left_layout.addWidget(buttons_group)
        left_layout.addWidget(metrics_group)
        left_layout.addStretch()

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.video_label, stretch=4)

        live_group = QGroupBox("Real-time metrics")
        live_grid = QGridLayout(live_group)
        for i, key in enumerate(LIVE_METRICS):
            name = key.replace("_", " ")
            value = QLabel("—")
            self.live_labels[key] = value
            live_grid.addWidget(QLabel(name), i // 2, (i % 2) * 2)
            live_grid.addWidget(value, i // 2, (i % 2) * 2 + 1)
        right_layout.addWidget(live_group)

        if pg is not None:
            plots = QWidget()
            plots_layout = QHBoxLayout(plots)
            self.speed_plot = pg.PlotWidget(title="Speed vs time")
            self.speed_plot.setLabel("left", "Speed", units="um/s")
            self.speed_plot.setLabel("bottom", "Time", units="s")
            self.speed_curve = self.speed_plot.plot([], [])

            self.turn_plot = pg.PlotWidget(title="Mean absolute turn / alignment proxy")
            self.turn_plot.setLabel("left", "Value")
            self.turn_plot.setLabel("bottom", "Time", units="s")
            self.turn_curve = self.turn_plot.plot([], [])

            plots_layout.addWidget(self.speed_plot)
            plots_layout.addWidget(self.turn_plot)
            right_layout.addWidget(plots, stretch=2)
        else:
            right_layout.addWidget(QLabel("Install pyqtgraph for live plots: pip install pyqtgraph"))

        right_layout.addWidget(self.status_label)

        root.addWidget(left_scroll)
        root.addWidget(right, stretch=1)

    def connect_signals(self):
        self.browse_video_btn.clicked.connect(self.choose_video)
        self.browse_output_btn.clicked.connect(self.choose_output_dir)
        self.start_btn.clicked.connect(self.start_tracking)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.reselect_btn.clicked.connect(self.reselect_roi)
        self.select_all_btn.clicked.connect(lambda: self.set_all_metrics(True))
        self.select_none_btn.clicked.connect(lambda: self.set_all_metrics(False))

    def set_all_metrics(self, checked: bool):
        for cb in self.metric_checks.values():
            cb.setChecked(checked)

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose video file",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.tif *.tiff);;All files (*.*)",
        )
        if path:
            self.video_path_edit.setText(path)
            if not self.output_dir_edit.text():
                self.output_dir_edit.setText(os.path.dirname(path))

    def choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if path:
            self.output_dir_edit.setText(path)

    def read_config(self) -> AnalysisConfig:
        video_path = self.video_path_edit.text().strip()
        if not video_path:
            raise ValueError("Choose a video file first.")
        if not os.path.exists(video_path):
            raise ValueError("Video file does not exist.")

        return AnalysisConfig(
            video_path=video_path,
            display_scale=float(self.display_scale_spin.value()),
            microns_per_pixel=float(self.microns_spin.value()),
            reselect_exclusion_s=float(self.exclusion_spin.value()),
            analysis_duration_s=float(self.duration_spin.value()),
            field_angle_deg=float(self.field_angle_spin.value()),
            smoothing_window=int(self.window_spin.value()),
            output_dir=self.output_dir_edit.text().strip(),
        )

    def selected_summary_metrics(self) -> List[str]:
        return [key for key, cb in self.metric_checks.items() if cb.isChecked()]

    def start_tracking(self):
        try:
            cfg = self.read_config()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid setup", str(exc))
            return

        self.plot_times.clear()
        self.plot_speeds.clear()
        self.plot_turns.clear()
        if self.speed_curve is not None:
            self.speed_curve.setData([], [])
        if self.turn_curve is not None:
            self.turn_curve.setData([], [])

        self.worker = TrackerWorker(cfg, self.selected_summary_metrics())
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.metrics_ready.connect(self.update_metrics)
        self.worker.finished_ok.connect(self.finished_ok)
        self.worker.failed.connect(self.failed)
        self.worker.status_changed.connect(self.set_status)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.reselect_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.set_status("Starting...")

    def toggle_pause(self):
        if not self.worker:
            return
        paused = self.pause_btn.text() == "Pause"
        self.worker.set_paused(paused)
        self.pause_btn.setText("Resume" if paused else "Pause")
        self.set_status("Paused" if paused else "Running")

    def stop_tracking(self):
        if self.worker:
            self.worker.request_stop()
            self.set_status("Stopping after current frame...")

    def reselect_roi(self):
        if self.worker:
            self.worker.request_reselect()
            self.set_status("ROI reselection requested.")

    @Slot(np.ndarray)
    def update_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

    @Slot(dict)
    def update_metrics(self, metrics: dict):
        if "status" in metrics:
            self.set_status(str(metrics["status"]))

        for key, label in self.live_labels.items():
            value = metrics.get(key, None)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                label.setText("—")
            elif isinstance(value, float):
                label.setText(f"{value:.4g}")
            else:
                label.setText(str(value))

        t = metrics.get("time_s")
        speed = metrics.get("speed_smooth_um_s")
        turn = metrics.get("mean_abs_turn_deg")
        if t is not None and speed is not None and not np.isnan(speed):
            self.plot_times.append(float(t))
            self.plot_speeds.append(float(speed))
            if self.speed_curve is not None:
                self.speed_curve.setData(self.plot_times, self.plot_speeds)
        if t is not None and turn is not None and not np.isnan(turn):
            self.plot_turns.append(float(turn))
            if self.turn_curve is not None:
                self.turn_curve.setData(self.plot_times[-len(self.plot_turns):], self.plot_turns)

    @Slot(str, str, dict)
    def finished_ok(self, tracking_csv: str, summary_csv: str, summary: dict):
        self.reset_buttons()
        lines = [f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}" for k, v in summary.items()]
        QMessageBox.information(
            self,
            "Analysis complete",
            "Saved files:\n"
            f"{tracking_csv}\n"
            f"{summary_csv}\n\n"
            + "\n".join(lines),
        )
        self.set_status(f"Done. Saved: {summary_csv}")

    @Slot(str)
    def failed(self, message: str):
        self.reset_buttons()
        QMessageBox.critical(self, "Analysis failed", message)
        self.set_status(f"Failed: {message}")

    @Slot(str)
    def set_status(self, text: str):
        self.status_label.setText(text)

    def reset_buttons(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.reselect_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        self.worker = None

    def closeEvent(self, event):
        if self.worker:
            self.worker.request_stop()
            self.worker.wait(2000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MicroSwim()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
