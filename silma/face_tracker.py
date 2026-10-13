"""Kasvojen tunnistus OpenCV:n YuNet-mallilla ja kohdepisteen 3D-arvio."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot, tan, radians

import cv2
import numpy as np

from .config import CameraConfig, DetectorConfig


@dataclass
class Face:
    box: tuple  # (x, y, w, h) täysikokoisen kuvan pikseleinä
    eyes_mid: tuple  # silmien välinen piste (u, v), pikseleinä
    interocular_px: float
    score: float


@dataclass
class Target:
    """Katseen kohde lelun koordinaatistossa (ks. config.py)."""

    position_m: tuple
    yaw: float
    pitch: float


class FaceDetector:
    def __init__(self, cfg: DetectorConfig):
        if not cfg.model_path.exists():
            raise FileNotFoundError(f"YuNet-mallia ei löydy: {cfg.model_path}")
        self.cfg = cfg
        self._net = cv2.FaceDetectorYN.create(
            str(cfg.model_path), "", (320, 240), cfg.score_threshold
        )
        self._input_size = None

    def detect(self, frame_bgr) -> list:
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.cfg.detect_width / w)
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale) if scale < 1.0 else frame_bgr
        size = (small.shape[1], small.shape[0])
        if size != self._input_size:
            self._net.setInputSize(size)
            self._input_size = size
        _, rows = self._net.detect(small)
        if rows is None:
            return []
        faces = []
        for r in rows / np.array([scale] * 14 + [1.0]):
            # YuNet: x, y, w, h, oikea silmä (x, y), vasen silmä (x, y),
            # nenä, suupielet, luotettavuus.
            re, le = r[4:6], r[6:8]
            faces.append(
                Face(
                    box=tuple(r[0:4]),
                    eyes_mid=tuple((re + le) / 2),
                    interocular_px=float(np.linalg.norm(re - le)),
                    score=float(r[14]),
                )
            )
        return faces


def select_face(faces: list, previous: Face | None) -> Face | None:
    """Pysy samoissa kasvoissa, jos ne löytyvät läheltä; muuten valitse suurin."""
    if not faces:
        return None
    if previous is not None:
        px, py = previous.eyes_mid
        nearest = min(faces, key=lambda f: hypot(f.eyes_mid[0] - px, f.eyes_mid[1] - py))
        if hypot(nearest.eyes_mid[0] - px, nearest.eyes_mid[1] - py) < previous.box[2]:
            return nearest
    return max(faces, key=lambda f: f.box[2] * f.box[3])


def face_to_target(
    face: Face, frame_size: tuple, cam: CameraConfig, interocular_m: float
) -> Target:
    """Muunna kuvapiste 3D-pisteeksi ja edelleen silmän kääntökulmiksi.

    Kamera on eri kohdassa kuin silmä (yleensä sen yläpuolella), joten
    pelkkä kuvakulma ei riitä: lähellä olevat kasvot näkyvät kamerasta eri
    kulmassa kuin silmästä. Etäisyys arvioidaan silmien välisestä
    pikselietäisyydestä (pinhole-malli), ja piste siirretään silmän
    origoon. Tämä on karkea arvio (n. ±20 %), mutta parallaksikorjaukseen riittävä.
    """
    w, h = frame_size
    f = (w / 2) / tan(radians(cam.hfov_deg) / 2)
    u, v = face.eyes_mid
    z = f * interocular_m / max(face.interocular_px, 1.0)
    x_img = (u - w / 2) * z / f  # kameran oikealle
    y_img = -(v - h / 2) * z / f  # ylös
    # Kamera katsoo lelun kanssa samaan suuntaan: kameran oikea = lelun
    # oikea = katsojan vasen = -x. Peilattu kuva kääntää tämän.
    x = x_img if cam.mirrored else -x_img
    ox, oy, oz = cam.offset_m
    p = (x + ox, y_img + oy, z + oz)
    yaw = atan2(p[0], p[2])
    pitch = atan2(p[1], hypot(p[0], p[2]))
    return Target(position_m=p, yaw=yaw, pitch=pitch)
