import math

def sample_frame_indices(
    frame_count: int,
    duration_seconds: float,
    segment_start_seconds: float,
    segment_end_seconds: float,
    view_id: str,
) -> list[int]:
    if frame_count <= 0 or duration_seconds <= 0:
        raise ValueError("video frame count and duration must be positive")
    start = max(0.0, float(segment_start_seconds))
    end = min(float(duration_seconds), float(segment_end_seconds))
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError(f"invalid full-video bound [{start}, {end}]")
    first = max(0, math.ceil(start * frame_count / duration_seconds - 0.5))
    last = min(frame_count - 1, math.floor(end * frame_count / duration_seconds - 0.5))
    if last < first:
        first = last = min(frame_count - 1, max(0, frame_count // 2))
    size = last - first + 1

    def uniform(count: int, phase: bool = False) -> list[int]:
        if phase:
            positions = [min(size - 1, int((index + 0.5) * size / count)) for index in range(count)]
        elif count == 1:
            positions = [size // 2]
        else:
            positions = [round(index * (size - 1) / (count - 1)) for index in range(count)]
        return [first + position for position in positions]

    if view_id == "base":
        return uniform(16)
    if view_id == "invariant_nuisance":
        return uniform(8, phase=True)
    if view_id == "dense_acquisition":
        return uniform(24)
    if view_id == "temporal_order_destruction":
        return list(reversed(uniform(8)))
    raise ValueError(f"unknown HERBench view {view_id!r}")

VIEW_IDS={0:"base",2:"invariant_nuisance",4:"dense_acquisition",6:"temporal_order_destruction"}
FRAME_COUNTS={0:16,2:8,4:24,6:8}
def indices(row,frames,duration,phase):
    return sample_frame_indices(frames,duration,float(row["start"]),float(row["end"]),VIEW_IDS[phase])
