from math import atan2, degrees, radians

import numpy as np
import pytest

from silma.behavior import EyeController, EyeState, lid_angles
from silma.config import CameraConfig, DetectorConfig, EyeConfig, LidConfig
from silma.face_tracker import Face, FaceDetector, face_to_target
from silma.renderer import EyeRenderer
from silma.servos import ServoLayout, servo_angles


def _face_at(u, v, iod_px=60):
    return Face(box=(u - 50, v - 60, 100, 120), eyes_mid=(u, v), interocular_px=iod_px, score=0.9)


def test_centered_face_corrects_camera_parallax():
    cam = CameraConfig(offset_m=(0.0, 0.05, 0.0))
    t = face_to_target(_face_at(320, 240), (640, 480), cam, 0.063)
    z = t.position_m[2]
    assert t.yaw == pytest.approx(0.0, abs=1e-9)
    # Kamera on silmän yläpuolella -> silmän pitää katsoa hieman ylös.
    assert t.pitch == pytest.approx(atan2(0.05, z), rel=1e-6)


def test_face_on_image_right_turns_eye_toward_negative_x():
    cam = CameraConfig()
    t = face_to_target(_face_at(600, 240), (640, 480), cam, 0.063)
    assert t.yaw < 0
    cam.mirrored = True
    assert face_to_target(_face_at(600, 240), (640, 480), cam, 0.063).yaw > 0


def test_closer_face_gives_shorter_distance():
    cam = CameraConfig()
    near = face_to_target(_face_at(320, 240, 120), (640, 480), cam, 0.063)
    far = face_to_target(_face_at(320, 240, 40), (640, 480), cam, 0.063)
    assert near.position_m[2] < far.position_m[2]


def _blink_starts(states):
    lids = np.array([s.lid_open for s in states])
    return np.sum((lids[1:] < 0.5) & (lids[:-1] >= 0.5))


def test_blink_rate_is_human_like():
    ctrl = EyeController(EyeConfig(), seed=1)
    dt = 1 / 60
    states = [ctrl.update((0.0, 0.0), dt) for _ in range(int(600 / dt))]
    per_min = _blink_starts(states) / 10
    assert 12 <= per_min <= 25


def test_gaze_reaches_target_and_respects_limits():
    cfg = EyeConfig()
    ctrl = EyeController(cfg, seed=2)
    for _ in range(60):
        s = ctrl.update((radians(20), radians(-10)), 1 / 60)
    assert degrees(s.yaw) == pytest.approx(20, abs=1.0)
    assert degrees(s.pitch) == pytest.approx(-10, abs=1.0)
    for _ in range(60):
        s = ctrl.update((radians(80), 0.0), 1 / 60)
    assert s.yaw <= cfg.gaze.max_yaw + radians(1)


def test_saccade_is_fast():
    ctrl = EyeController(EyeConfig(), seed=3)
    ctrl.update((0.0, 0.0), 1 / 60)
    t = 0.0
    while abs(degrees(ctrl.update((radians(20), 0.0), 1 / 240).yaw) - 20) > 1.0:
        t += 1 / 240
        assert t < 0.2


def test_lost_face_makes_eye_wander():
    ctrl = EyeController(EyeConfig(), seed=4)
    for _ in range(30):
        ctrl.update((0.0, 0.0), 1 / 30)
    yaws = [ctrl.update(None, 1 / 30).yaw for _ in range(300)]
    assert np.ptp(yaws) > radians(3)


def _pupil_centroid_x(img):
    n = img.shape[0]
    yy, xx = np.mgrid[:n, :n]
    on_ball = (xx - n / 2) ** 2 + (yy - n / 2) ** 2 < (0.25 * n) ** 2
    dark = (img.max(axis=2) < 60) & on_ball
    return np.nonzero(dark)[1].mean()


def test_renderer_pupil_follows_yaw_and_lids_close():
    r = EyeRenderer(200)
    up, lo = lid_angles(LidConfig(), 1.0, 0.0)
    center = _pupil_centroid_x(r.render(EyeState(upper_lid=up, lower_lid=lo)))
    right = _pupil_centroid_x(r.render(EyeState(yaw=radians(25), upper_lid=up, lower_lid=lo)))
    assert right > center + 10

    up, lo = lid_angles(LidConfig(), 0.0, 0.0)
    closed = r.render(EyeState(upper_lid=up, lower_lid=lo, lid_open=0.0))
    assert np.sum(closed.min(axis=2) > 200) < 50  # ei valkoista näkyvissä


def test_servo_mapping_clamps():
    layout = ServoLayout()
    a = servo_angles(EyeState(yaw=radians(200)), layout)
    assert a[layout.yaw.channel] == layout.yaw.max_deg


def test_detector_runs_on_empty_frame():
    det = FaceDetector(DetectorConfig())
    assert det.detect(np.zeros((480, 640, 3), np.uint8)) == []
