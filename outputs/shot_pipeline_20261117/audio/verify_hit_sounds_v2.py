"""Focused synthetic checks; no match audio or held-out labels are loaded.

Run from the repository root:
study/.venv/Scripts/python.exe -B -X utf8 outputs/shot_pipeline_20261117/audio/verify_hit_sounds_v2.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import detect_hit_sounds as base
import detect_hit_sounds_v2 as v2


def candidates(frames, point=7, rally="A"):
    return pd.DataFrame({"candidate_id": [f"c{i:04d}" for i in range(len(frames))],
                         "point": [point] * len(frames), "rally": [rally] * len(frames),
                         "candidate_frame": frames})


def check_review_windows():
    segment = base.Segment(7, "A", 100, 200, pd.DataFrame())
    # [100,110], [100,120], [105,125], [185,200]: sum65, union40 frames.
    data = candidates([100, 110, 115, 195])
    burden = v2.window_burden([segment], data)
    np.testing.assert_allclose(burden["review_window_sum_sec"], 65 * base.FRAME_SEC)
    np.testing.assert_allclose(burden["review_time_ratio_plusminus10_frames"], .65)
    np.testing.assert_allclose(base.review_coverage([segment], data), .40)
    np.testing.assert_allclose(v2.window_burden([segment], data.iloc[:0])[
        "review_time_ratio_plusminus10_frames"], 0)
    np.testing.assert_allclose(base.review_coverage([segment], data.iloc[:0]), 0)
    # Windows are assigned by point AND rally, not by source-frame coincidence.
    other = candidates([150], rally="B")
    np.testing.assert_allclose(v2.window_burden([segment], other)[
        "review_time_ratio_plusminus10_frames"], 0)
    np.testing.assert_allclose(base.review_coverage([segment], other), 0)
    # The summed metric can exceed100%, while union remains bounded.
    overlapping = candidates([110, 120, 130, 140, 150, 160, 170, 180, 190])
    np.testing.assert_allclose(v2.window_burden([segment], overlapping)[
        "review_time_ratio_plusminus10_frames"], 1.8)
    np.testing.assert_allclose(base.review_coverage([segment], overlapping), 1.)


def check_candidate_targets():
    labels_a = pd.DataFrame({"point": [7, 7], "rally": ["A", "A"],
                             "contact_frame": [150, 190], "confidence": ["clear", "inferred"]})
    labels_b = pd.DataFrame({"point": [7], "rally": ["B"],
                             "contact_frame": [169], "confidence": ["clear"]})
    segments = [base.Segment(7, "A", 100, 230, labels_a),
                base.Segment(7, "B", 300, 430, labels_b)]
    data = candidates([139, 140, 149, 150, 160, 161, 169, 180, 190, 200, 201])
    truth, nearest = v2.candidate_targets(segments, data)
    # More than one candidate can be positive for the same label; +/-10 is inclusive.
    np.testing.assert_array_equal(truth, [False, True, True, True, True, False,
                                         False, True, True, True, False])
    np.testing.assert_array_equal(nearest, [11, 10, 1, 0, 10, 11, 19, 10, 0, 10, 11])


def synthetic_features():
    rate = base.SAMPLE_RATE
    t = np.arange(2 * rate) / rate
    rng = np.random.default_rng(17)
    floor = 1e-4 * rng.standard_normal(len(t))
    click = floor + .25 * rng.standard_normal(len(t)) * np.exp(-.5 * ((t - 1.) / .0015) ** 2)
    # A sustained harmonic burst, with a central120ms plateau. Its50ms-after
    # energy stays high; a4ms broadband click should already have decayed.
    envelope = np.zeros_like(t)
    core = np.abs(t - 1.) <= .06
    edge = (np.abs(t - 1.) > .06) & (np.abs(t - 1.) < .08)
    envelope[core] = 1.
    envelope[edge] = .5 * (1 + np.cos(np.pi * (np.abs(t[edge] - 1.) - .06) / .02))
    tone = floor + envelope * sum(.04 * np.sin(2 * np.pi * f * t)
                                  for f in (1000, 2000, 4000, 6000))
    rows = []
    for name, audio in (("brief_broadband", click), ("sustained_harmonic", tone)):
        starts = np.arange(0, len(audio) - base.ENERGY_SAMPLES + 1, base.HOP_SAMPLES)
        frames = (starts + (base.ENERGY_SAMPLES - 1) / 2) / rate / base.FRAME_SEC
        segment = base.Segment(7, "A", 0, int(2 / base.FRAME_SEC), pd.DataFrame(),
                               audio=audio, audio_start_sec=0., feature_frames=frames)
        # Choose a peak in the central10ms, rather than assuming an envelope
        # sample coincides exactly with the energy window's centre.
        power, _, _ = v2.band_envelope(segment, (4000, 12000))
        near = np.flatnonzero(np.abs(frames * base.FRAME_SEC - 1.) <= .005)
        index = near[np.argmax(power[near])]
        frame_float = float(frames[index])
        data = candidates([int(np.floor(frame_float + .5))])
        data["candidate_frame_float"] = frame_float
        data["candidate_time_sec"] = frame_float * base.FRAME_SEC
        row = v2.add_features([segment], data).iloc[0]
        rows.append(row)
    click_features, tone_features = rows
    assert click_features.half_width_ms < 10, click_features.half_width_ms
    assert tone_features.half_width_ms > 80, tone_features.half_width_ms
    assert click_features.post50_drop_ln > tone_features.post50_drop_ln + 4
    assert click_features.energy_concentration_20_100 > .9
    assert tone_features.energy_concentration_20_100 < .5
    assert click_features.low_rise_ln > 0 and click_features.high_rise_ln > 0
    for row in rows:
        assert 0 <= row.spectral_flatness_100ms <= 1
        assert abs(row.low_peak_lag_within4_ms) <= 4
    return pd.DataFrame(rows).assign(signal=["brief_broadband", "sustained_harmonic"])[
        ["signal", "half_width_ms", "post50_drop_ln", "energy_concentration_20_100",
         "low_rise_ln", "high_rise_ln", "spectral_flatness_100ms"]]


def main():
    check_review_windows()
    check_candidate_targets()
    features = synthetic_features()
    print(features.to_string(index=False))
    print("PASS: clipped sum/union, empty and boundary windows, train-target tolerance, and synthetic transient features")


if __name__ == "__main__":
    main()
