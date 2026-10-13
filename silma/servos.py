"""EyeState -> servokulmat. Fyysinen lelu käyttää tätä simulaattorin sijaan.

HUOM: tätä ei ole vielä testattu oikealla raudalla. Kalibrointiarvot
(keskikohta, suunta, välitys) määräytyvät mekanismin mukaan ja on
mitattava, kun osat on tulostettu. Oletus: PCA9685-servo-ohjain I2C:llä
(`pip install adafruit-circuitpython-servokit`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import degrees


@dataclass
class ServoChannel:
    channel: int
    center_deg: float = 90.0
    gain: float = 1.0  # servoasteita per silmäaste (vipuvarsien välitys)
    invert: bool = False
    min_deg: float = 0.0
    max_deg: float = 180.0

    def angle(self, value_deg: float) -> float:
        v = -value_deg if self.invert else value_deg
        return max(self.min_deg, min(self.max_deg, self.center_deg + self.gain * v))


@dataclass
class ServoLayout:
    yaw: ServoChannel = field(default_factory=lambda: ServoChannel(0))
    pitch: ServoChannel = field(default_factory=lambda: ServoChannel(1))
    upper_lid: ServoChannel = field(default_factory=lambda: ServoChannel(2))
    lower_lid: ServoChannel = field(default_factory=lambda: ServoChannel(3, invert=True))


def servo_angles(state, layout: ServoLayout) -> dict:
    return {
        layout.yaw.channel: layout.yaw.angle(degrees(state.yaw)),
        layout.pitch.channel: layout.pitch.angle(degrees(state.pitch)),
        layout.upper_lid.channel: layout.upper_lid.angle(degrees(state.upper_lid)),
        layout.lower_lid.channel: layout.lower_lid.angle(degrees(state.lower_lid)),
    }


class ServoOutput:
    def __init__(self, layout: ServoLayout | None = None):
        from adafruit_servokit import ServoKit  # vain Raspberry Pi:llä

        self.layout = layout or ServoLayout()
        self.kit = ServoKit(channels=16)

    def apply(self, state) -> None:
        for ch, ang in servo_angles(state, self.layout).items():
            self.kit.servo[ch].angle = ang
