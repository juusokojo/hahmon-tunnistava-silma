"""Kohteen valinta: ensin kameraan katsovat kasvot, sitten muut kasvot,
sitten ihmishahmot.

Hahmontunnistus ajetaan vain, kun kasvoja ei löydy, koska se on
kasvontunnistusta raskaampi (merkitsevää Raspberry Pi:llä).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .config import EyeConfig
from .face_tracker import FaceDetector, Target, face_to_target, facing_camera, focal_px, point_to_target

MODE_FACING = "kasvot, katsoo"
MODE_FACE = "kasvot, sivulta"
MODE_PERSON = "hahmo"


@dataclass
class Detection:
    mode: str
    box: tuple  # (x, y, w, h)
    point: tuple  # kuvapiste, johon silmä katsoo
    target: Target


def _center(box):
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def _area(box):
    return box[2] * box[3]


def pick(candidates: list, previous: Detection | None, box_of, takeover: float = 2.5) -> object | None:
    """Pysy samassa kohteessa, jos se löytyy läheltä; muuten valitse suurin.

    Selvästi isompi (= lähempänä oleva) kohde vie kuitenkin huomion, vaikka
    edellinen olisi yhä näkyvissä.
    """
    if not candidates:
        return None
    largest = max(candidates, key=lambda c: _area(box_of(c)))
    if previous is not None:
        px, py = _center(previous.box)
        nearest = min(candidates, key=lambda c: hypot(_center(box_of(c))[0] - px, _center(box_of(c))[1] - py))
        cx, cy = _center(box_of(nearest))
        same = hypot(cx - px, cy - py) < max(previous.box[2], 40)
        if same and _area(box_of(largest)) < takeover * _area(box_of(nearest)):
            return nearest
    return largest


def person_head_point(box: tuple, head_fraction: float) -> tuple:
    x, y, w, h = box
    return x + w / 2, y + h * head_fraction


def person_distance(box: tuple, frame_size: tuple, f: float, height_m: float, width_m: float) -> float:
    """Etäisyys hahmon koosta. Jos hahmo on rajautunut kuvan reunaan
    (esim. lähellä kameraa vain ylävartalo näkyy), pituutta ei voi käyttää."""
    x, y, w, h = box
    fully_visible = y > 2 and y + h < frame_size[1] - 2
    z = f * height_m / h if fully_visible else f * width_m / max(w, 1.0)
    return min(max(z, 0.4), 10.0)


class Tracker:
    def __init__(self, cfg: EyeConfig):
        self.cfg = cfg
        self.faces = FaceDetector(cfg.detector)
        self.people = None
        if cfg.person.enabled:
            from .person_detector import PersonDetector

            self.people = PersonDetector(cfg.person)
        self.current: Detection | None = None

    def update(self, frame) -> Detection | None:
        h, w = frame.shape[:2]
        size = (w, h)
        cam, det = self.cfg.camera, self.cfg.detector
        prev = self.current if self.current else None

        faces = self.faces.detect(frame)
        facing = [f for f in faces if facing_camera(f, det.facing_max_offset)]
        for mode, group in ((MODE_FACING, facing), (MODE_FACE, faces)):
            face = pick(group, prev, lambda f: f.box)
            if face is not None:
                self.current = Detection(
                    mode, face.box, face.eyes_mid, face_to_target(face, size, cam, det.interocular_m)
                )
                return self.current

        if self.people is not None:
            pc = self.cfg.person
            person = pick(self.people.detect(frame), prev, lambda p: p.box)
            if person is not None:
                point = person_head_point(person.box, pc.head_fraction)
                z = person_distance(person.box, size, focal_px(w, cam), pc.person_height_m, pc.person_width_m)
                self.current = Detection(MODE_PERSON, person.box, point, point_to_target(point, z, size, cam))
                return self.current

        self.current = None
        return None
