"""Silmän 3D-simulaattori ilman grafiikkakirjastoja.

Malli vastaa tulostettavaa mekanismia: silmämuna (pallo, r = 1) pyörii
kahden akselin varassa, ja kaksi luomikuorta (pallonkuoret, r = LID_R)
kiertyvät vaaka-akselin ympäri. Koska kaikki pinnat ovat palloja ja
kamera on paikallaan, säteiden osumapisteet lasketaan kerran
(`set_view`), ja jokainen kuva on pelkkiä numpy-pistetuloja -> nopea
myös Raspberry Pi:llä.
"""

from math import cos, radians, sin

import numpy as np

from .behavior import EyeState

LID_R = 1.07
PUPIL_ANGLE = radians(13)  # mustan pisteen kulmasäde
IRIS_ANGLE = radians(0)  # 0 = ei iiristä (pelkkä musta piste kuten lelussa)

SCLERA = np.array([235, 238, 240], np.float32)  # BGR
PUPIL = np.array([20, 20, 20], np.float32)
IRIS = np.array([150, 110, 60], np.float32)
LID = np.array([120, 170, 225], np.float32)
BG_TOP = np.array([60, 50, 45], np.float32)
BG_BOTTOM = np.array([25, 22, 20], np.float32)

LIGHT = np.array([-0.4, 0.6, 0.7])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def _sphere_hit(origin, dirs, r):
    """Säteen ja origokeskisen pallon lähin etupuolen osuma."""
    b = dirs @ origin
    c = origin @ origin - r * r
    disc = b * b - c
    hit = disc > 0
    t = -b - np.sqrt(np.where(hit, disc, 0))
    return hit & (t > 0), origin + dirs * t[..., None]


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def gaze_matrix(yaw: float, pitch: float) -> np.ndarray:
    """Rotaatio, joka vie silmän paikallisen +z-akselin katsesuuntaan."""
    cy, sy, cp, sp = cos(yaw), sin(yaw), cos(-pitch), sin(-pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return ry @ rx


class EyeRenderer:
    def __init__(self, size: int = 400, distance: float = 6.0, fov_deg: float = 30.0):
        self.size = size
        self.distance = distance
        self.fov = radians(fov_deg)
        self.set_view(0.0, 0.0)

    def set_view(self, view_yaw: float, view_pitch: float) -> None:
        """Katselukulma (kameran kierto silmän ympäri) testausta varten."""
        n = self.size
        cam_dir = np.array(
            [sin(view_yaw) * cos(view_pitch), sin(view_pitch), cos(view_yaw) * cos(view_pitch)]
        )
        origin = cam_dir * self.distance
        forward = -cam_dir
        right = np.cross(forward, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        half = np.tan(self.fov / 2)
        s = np.linspace(-half, half, n)
        px, py = np.meshgrid(s, -s)
        dirs = forward + px[..., None] * right + py[..., None] * up
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

        self._eye_hit, eye_p = _sphere_hit(origin, dirs, 1.0)
        self._lid_hit, lid_p = _sphere_hit(origin, dirs, LID_R)
        self._eye_p = eye_p.astype(np.float32)
        # Luomikuoren elevaatiokulma x-akselin ympäri (0 = suoraan eteen).
        self._lid_phi = np.arctan2(lid_p[..., 1], lid_p[..., 2]).astype(np.float32)

        view = -dirs
        self._eye_shade = self._shade(eye_p / 1.0, view, 0.35, 60)
        self._lid_shade = self._shade(lid_p / LID_R, view, 0.25, 20)
        grad = np.linspace(0, 1, n, dtype=np.float32)[:, None, None]
        self._bg = (BG_TOP * (1 - grad) + BG_BOTTOM * grad) * np.ones((1, n, 1), np.float32)

    @staticmethod
    def _shade(normal, view, spec_strength, shininess):
        diffuse = np.clip(normal @ LIGHT, 0, 1)
        half = LIGHT + view
        half /= np.linalg.norm(half, axis=-1, keepdims=True)
        spec = np.clip(np.sum(normal * half, axis=-1), 0, 1) ** shininess
        return (
            (0.35 + 0.65 * diffuse)[..., None].astype(np.float32),
            (spec_strength * 255 * spec)[..., None].astype(np.float32),
        )

    def render(self, state: EyeState) -> np.ndarray:
        img = self._bg.copy()

        # Silmämuna: paikalliset koordinaatit -> kulma pupillin akselista.
        local = self._eye_p @ gaze_matrix(state.yaw, state.pitch).astype(np.float32)
        cos_a = np.clip(local[..., 2], -1, 1)
        ang = np.arccos(cos_a)
        aa = radians(0.8)  # reunan pehmennys (antialiasointi)
        color = np.broadcast_to(SCLERA, img.shape).copy()
        if IRIS_ANGLE > 0:
            k = _smoothstep(IRIS_ANGLE + aa, IRIS_ANGLE - aa, ang)[..., None]
            color = color * (1 - k) + IRIS * k
        k = _smoothstep(PUPIL_ANGLE + aa, PUPIL_ANGLE - aa, ang)[..., None]
        color = color * (1 - k) + PUPIL * k
        diffuse, spec = self._eye_shade
        eye_rgb = color * diffuse + spec
        m = self._eye_hit
        img[m] = eye_rgb[m]

        # Luomikuoret: peittävät, jos elevaatio on reunakulman ulkopuolella.
        phi = self._lid_phi
        cover_up = _smoothstep(state.upper_lid - aa, state.upper_lid + aa, phi)
        cover_lo = _smoothstep(state.lower_lid + aa, state.lower_lid - aa, phi)
        cover = np.maximum(cover_up, cover_lo) * self._lid_hit
        # Tumma reunaviiva antaa luomelle paksuuden tunnun.
        edge_w = radians(2.5)
        rim = np.maximum(
            _smoothstep(state.upper_lid + edge_w, state.upper_lid, phi) * cover_up,
            _smoothstep(state.lower_lid - edge_w, state.lower_lid, phi) * cover_lo,
        )[..., None]
        diffuse, spec = self._lid_shade
        lid_rgb = LID * diffuse * (1 - 0.45 * rim) + spec
        c = cover[..., None]
        img = img * (1 - c) + lid_rgb * c

        return np.clip(img, 0, 255).astype(np.uint8)
