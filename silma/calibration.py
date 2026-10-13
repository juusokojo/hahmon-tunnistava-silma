"""Katseen kalibrointi: "tästä kohdasta katsottuna silmä katsoo suoraan minuun".

Tasaisella näytöllä piirretty silmä näyttää katsovan katsojaa silloin, kun
pupilli on keskellä, katsojan paikasta riippumatta (ns. Mona Lisa -ilmiö).
Fyysisesti oikea kulma, esim. 15° ylös, kun kasvot ovat kameran akselin
yläpuolella, näyttää ruudulla siltä, että silmä katsoo katsojan ohi.
Kalibrointi tallentaa nykyisen kohteen kulman nollakohdaksi.

Oikeassa 3D-lelussa tätä ei tarvita (ilmiötä ei ole), mutta samalla voi
tarvittaessa korjata mekanismin pienen asennusvirheen.
"""

from __future__ import annotations

import json
from pathlib import Path

CALIBRATION_FILE = Path(__file__).resolve().parent.parent / ".kalibrointi.json"


def apply_calibration(target: tuple | None, offset: tuple) -> tuple | None:
    if target is None:
        return None
    return target[0] - offset[0], target[1] - offset[1]


def load_calibration(path: Path = CALIBRATION_FILE) -> tuple:
    try:
        data = json.loads(path.read_text())
        return float(data["yaw"]), float(data["pitch"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0, 0.0


def save_calibration(offset: tuple, path: Path = CALIBRATION_FILE) -> None:
    try:
        path.write_text(json.dumps({"yaw": offset[0], "pitch": offset[1]}))
    except OSError:
        pass
