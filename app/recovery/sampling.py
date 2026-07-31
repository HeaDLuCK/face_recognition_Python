def history_sample_interval_seconds(
    window_seconds: float,
    target_interval_seconds: float,
    max_frames: int,
) -> float:
    """Keep history sampling time-based while enforcing a per-job frame cap."""
    return max(
        float(target_interval_seconds),
        max(float(window_seconds), 1.0) / max(int(max_frames), 1),
    )
