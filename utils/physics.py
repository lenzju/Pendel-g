import numpy as np


def calculate_physics(states, fps, length):
    left_frames = []

    for frame, state in states:
        if "Links" in state:
            left_frames.append(frame)

    if len(left_frames) < 2:
        return 0, 0, 0

    filtered = [left_frames[0]]

    for f in left_frames[1:]:
        if f - filtered[-1] > fps * 0.5:
            filtered.append(f)

    if len(filtered) < 2:
        return 0, 0, 0

    periods = np.diff(filtered) / fps

    T = np.mean(periods)
    f = 1 / T

    g = (4 * np.pi**2 * length) / (T**2)

    return T, f, g
