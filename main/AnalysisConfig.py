from dataclasses import dataclass

@dataclass
class AnalysisConfig:
    video_path: str
    display_scale: float = 0.5
    microns_per_pixel: float = 1.0
    reselect_exclusion_s: float = 0.5
    analysis_duration_s: float = 60.0
    field_angle_deg: float = 0.0
    smoothing_window: int = 7
    output_dir: str = ""