from AnalysisConfig import AnalysisConfig
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

class MetricsEngine:
    def __init__(self, config: AnalysisConfig, fps: float):
        self.config = config
        self.fps = fps

    def add_motion_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cfg = self.config
        window = max(1, int(cfg.smoothing_window))

        df["x_um"] = df["corrected_x_px"] * cfg.microns_per_pixel
        df["y_um"] = df["corrected_y_px"] * cfg.microns_per_pixel

        df["dx_um"] = df["x_um"].diff()
        df["dy_um"] = df["y_um"].diff()
        df["dt_s"] = df["time_s"].diff()

        HEADING_STEP = max(2, window // 2)

        df["step_distance_um"] = np.sqrt(df["dx_um"] ** 2 + df["dy_um"] ** 2)
        df["speed_um_s"] = df["step_distance_um"] / df["dt_s"]

        df["heading_dx_um"] = (
            df["x_um"].shift(-HEADING_STEP) - df["x_um"].shift(HEADING_STEP)
        )

        df["heading_dy_um"] = (
            df["y_um"].shift(-HEADING_STEP) - df["y_um"].shift(HEADING_STEP)
        )

        df["heading_rad"] = np.arctan2(
            df["heading_dy_um"],
            df["heading_dx_um"],
        )
        heading_filled = df["heading_rad"].ffill().fillna(0)
        df["heading_unwrapped_rad"] = np.unwrap(heading_filled)

        df["turn_angle_deg"] = np.degrees(df["heading_unwrapped_rad"].diff())
        df["abs_turn_deg"] = df["turn_angle_deg"].abs()

        df["speed_smooth_um_s"] = (
            df["speed_um_s"].rolling(window, center=True, min_periods=1).median()
        )
        df["turn_smooth_deg"] = (
            df["turn_angle_deg"].rolling(window, center=True, min_periods=1).median()
        )
        df["abs_turn_smooth_deg"] = df["turn_smooth_deg"].abs()

        field_rad = np.deg2rad(cfg.field_angle_deg)
        df["velocity_parallel_field_um_s"] = (
            (df["dx_um"] * np.cos(field_rad) + df["dy_um"] * np.sin(field_rad))
            / df["dt_s"]
        )
        df["velocity_perpendicular_field_um_s"] = (
            (-df["dx_um"] * np.sin(field_rad) + df["dy_um"] * np.cos(field_rad))
            / df["dt_s"]
        )
        df["field_alignment"] = np.cos(df["heading_rad"] - field_rad)

        return df

    def mark_reselection_exclusions(
        self,
        df: pd.DataFrame,
        manual_reselect_frames: List[int],
    ) -> pd.DataFrame:
        df = df.copy()
        df["manual_reselect_nearby"] = False
        exclude_frames = int(self.config.reselect_exclusion_s * self.fps)

        for rf in manual_reselect_frames:
            df.loc[
                (df["frame"] >= rf - exclude_frames) & (df["frame"] <= rf),
                "manual_reselect_nearby",
            ] = True

        return df

    def valid_df(
        self,
        records: List[Dict],
        manual_reselect_frames: List[int],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame(records)
        if df.empty:
            return df, df
        df = self.add_motion_stats(df)
        df = self.mark_reselection_exclusions(df, manual_reselect_frames)
        valid = df[(df["tracking_success"]) & (~df["manual_reselect_nearby"])].copy()
        return df, valid

    def live_summary(
        self,
        records: List[Dict],
        manual_reselect_frames: List[int],
    ) -> Dict[str, float]:
        if len(records) < 5:
            return {"status": "Collecting data..."}

        df, valid = self.valid_df(records, manual_reselect_frames)
        if len(valid) < 3:
            return {"status": "Collecting valid tracking data..."}

        current = valid.iloc[-1]
        total_path = valid["step_distance_um"].sum()
        net_disp = np.sqrt(
            (valid["x_um"].iloc[-1] - valid["x_um"].iloc[0]) ** 2
            + (valid["y_um"].iloc[-1] - valid["y_um"].iloc[0]) ** 2
        )
        meander = net_disp / total_path if total_path > 0 else np.nan

        return {
            "status": "Tracking",
            "time_s": float(current["time_s"]),
            "speed_smooth_um_s": float(current["speed_smooth_um_s"]),
            "mean_speed_um_s": float(valid["speed_smooth_um_s"].mean()),
            "median_speed_um_s": float(valid["speed_smooth_um_s"].median()),
            "p95_speed_um_s": float(valid["speed_smooth_um_s"].quantile(0.95)),
            "total_path_um": float(total_path),
            "net_displacement_um": float(net_disp),
            "meandering_index": float(meander),
            "mean_abs_turn_deg": float(valid["abs_turn_smooth_deg"].mean()),
            "mean_velocity_parallel_field_um_s": float(
                valid["velocity_parallel_field_um_s"].mean()
            ),
            "mean_field_alignment": float(valid["field_alignment"].mean()),
            "excluded_frames": int(df["manual_reselect_nearby"].sum()),
        }

    def final_summary(
        self,
        records: List[Dict],
        manual_reselect_frames: List[int],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df, valid = self.valid_df(records, manual_reselect_frames)
        if len(valid) < 3:
            raise RuntimeError("Not enough valid frames for summary statistics.")

        total_path_um = valid["step_distance_um"].sum()
        net_displacement_um = np.sqrt(
            (valid["x_um"].iloc[-1] - valid["x_um"].iloc[0]) ** 2
            + (valid["y_um"].iloc[-1] - valid["y_um"].iloc[0]) ** 2
        )
        duration_s = valid["time_s"].iloc[-1] - valid["time_s"].iloc[0]
        meandering_index = (
            net_displacement_um / total_path_um if total_path_um > 0 else np.nan
        )

        summary = {
            "duration_s": duration_s,
            "mean_speed_um_s": valid["speed_smooth_um_s"].mean(),
            "median_speed_um_s": valid["speed_smooth_um_s"].median(),
            "std_speed_um_s": valid["speed_smooth_um_s"].std(),
            "p95_speed_um_s": valid["speed_smooth_um_s"].quantile(0.95),
            "total_path_um": total_path_um,
            "net_displacement_um": net_displacement_um,
            "persistence_length_proxy": net_displacement_um / duration_s if duration_s > 0 else np.nan,
            "meandering_index": meandering_index,
            "median_turn_deg_per_s": (valid["abs_turn_smooth_deg"] / valid["dt_s"]).median(),
            "mean_velocity_parallel_field_um_s": valid["velocity_parallel_field_um_s"].mean(),
            "abs_velocity_parallel_field_um_s": valid["velocity_parallel_field_um_s"].abs().mean(),
            "mean_velocity_perpendicular_field_um_s": valid["velocity_perpendicular_field_um_s"].mean(),
            "abs_velocity_perpendicular_field_um_s": valid["velocity_perpendicular_field_um_s"].abs().mean(),
            "mean_field_alignment": valid["field_alignment"].mean(),
            "std_field_alignment": valid["field_alignment"].std(),
            "manual_reselect_events": len(manual_reselect_frames),
            "excluded_reselect_frames": int(df["manual_reselect_nearby"].sum()),
            "valid_frames_used": len(valid),
        }

        return df, pd.DataFrame([summary])

