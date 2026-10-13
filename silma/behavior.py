"""Katseen ja luomien käyttäytyminen: sakkadit, seuranta, vilkuilu, räpäytykset.

Tämä moduuli ei tiedä mitään kamerasta tai moottoreista. Se saa
kohdekulman (tai None, jos kasvoja ei näy) ja aika-askeleen, ja tuottaa
EyeState-tilan, jonka simulaattori tai servo-ohjain toteuttaa.
"""

import random
from dataclasses import dataclass
from math import degrees, exp, hypot, log

from .config import BlinkConfig, EyeConfig, GazeConfig, LidConfig


@dataclass
class EyeState:
    yaw: float = 0.0
    pitch: float = 0.0
    upper_lid: float = 0.0  # reunan kulma, ks. LidConfig
    lower_lid: float = 0.0
    lid_open: float = 1.0  # 0 = kiinni, 1 = auki (ennen katseen seurantaa)
    tracking: bool = False


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _min_jerk(s: float) -> float:
    """Sileä 0→1-käyrä (minimum jerk), lähellä ihmissilmän sakkadin profiilia."""
    s = _clamp(s, 0.0, 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


class GazeController:
    def __init__(self, cfg: GazeConfig, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self.yaw = 0.0
        self.pitch = 0.0
        self._fix = (0.0, 0.0)  # nykyinen fiksaatiopiste
        self._sacc = None  # (alku, loppu, kesto, kulunut)
        self._since_seen = 1e9
        self._last_seen = (0.0, 0.0)
        self._next_idle = 0.0
        self._next_micro = self._uniform(cfg.microsaccade_interval_s)
        self.last_saccade_amp = 0.0  # luetaan räpäytyslogiikassa

    def _uniform(self, rng_range):
        return self.rng.uniform(*rng_range)

    def _clamp_angles(self, yaw, pitch):
        c = self.cfg
        return _clamp(yaw, -c.max_yaw, c.max_yaw), _clamp(pitch, -c.max_pitch, c.max_pitch)

    def _start_saccade(self, target):
        amp = hypot(target[0] - self.yaw, target[1] - self.pitch)
        dur = self.cfg.saccade_base_s + self.cfg.saccade_per_deg_s * degrees(amp)
        self._sacc = [(self.yaw, self.pitch), target, dur, 0.0]
        self._fix = target
        self.last_saccade_amp = amp

    def update(self, target: tuple | None, dt: float) -> None:
        c = self.cfg
        self.last_saccade_amp = 0.0

        if target is not None:
            target = self._clamp_angles(*target)
            self._since_seen = 0.0
            self._last_seen = target
        else:
            self._since_seen += dt

        # Päätä, mihin halutaan katsoa.
        if target is not None:
            desired = target
        elif self._since_seen < c.lost_hold_s:
            desired = self._last_seen
        else:
            self._next_idle -= dt
            if self._next_idle <= 0:
                self._next_idle = self._uniform(c.idle_glance_interval_s)
                # Painota keskelle ja välillä takaisin viimeiseen kohtaan.
                if self.rng.random() < 0.25:
                    desired = self._last_seen
                else:
                    desired = self._clamp_angles(
                        self.rng.gauss(0, c.idle_range / 2), self.rng.gauss(0, c.idle_range / 3)
                    )
                self._start_saccade(desired)
            desired = self._fix

        if self._sacc is None:
            err = hypot(desired[0] - self._fix[0], desired[1] - self._fix[1])
            if err > c.saccade_threshold:
                self._start_saccade(desired)
            else:
                # Hidas seuranta pienille liikkeille.
                k = 1 - exp(-c.pursuit_rate * dt)
                self._fix = (
                    self._fix[0] + (desired[0] - self._fix[0]) * k,
                    self._fix[1] + (desired[1] - self._fix[1]) * k,
                )
                self._next_micro -= dt
                if self._next_micro <= 0:
                    self._next_micro = self._uniform(c.microsaccade_interval_s)
                    a = c.microsaccade_amp
                    jitter = (self.rng.uniform(-a, a), self.rng.uniform(-a, a))
                    self._sacc = [
                        (self.yaw, self.pitch),
                        (self._fix[0] + jitter[0], self._fix[1] + jitter[1]),
                        0.02,
                        0.0,
                    ]

        if self._sacc is not None:
            start, end, dur, t = self._sacc
            t += dt
            s = _min_jerk(t / dur)
            self.yaw = start[0] + (end[0] - start[0]) * s
            self.pitch = start[1] + (end[1] - start[1]) * s
            self._sacc[3] = t
            if t >= dur:
                self._sacc = None
        else:
            self.yaw, self.pitch = self._fix


class BlinkController:
    def __init__(self, cfg: BlinkConfig, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self._t = 0.0
        self._phase_t = None  # aika räpäytyksen alusta, None = ei käynnissä
        self._pending_double = False
        self._next = self._draw_interval()

    def _draw_interval(self) -> float:
        c = self.cfg
        # Log-normaali, jonka keskiarvo on mean_interval_s.
        mu = c.mean_interval_s
        x = self.rng.lognormvariate(log(mu) - c.interval_sigma**2 / 2, c.interval_sigma)
        return max(c.min_interval_s, x)

    @property
    def blinking(self) -> bool:
        return self._phase_t is not None

    def trigger(self) -> None:
        if self._phase_t is None:
            self._phase_t = 0.0

    def update(self, dt: float, saccade_amp: float = 0.0) -> float:
        """Palauttaa luomen avautumisen 0..1."""
        c = self.cfg
        if (
            saccade_amp >= c.saccade_blink_min_amp
            and not self.blinking
            and self.rng.random() < c.saccade_blink_prob
        ):
            self.trigger()

        if self._phase_t is None:
            self._t += dt
            if self._t >= self._next:
                self.trigger()
            return 1.0

        self._phase_t += dt
        t = self._phase_t
        total = c.close_s + c.hold_s + c.open_s
        if t < c.close_s:
            openness = 1 - _min_jerk(t / c.close_s)
        elif t < c.close_s + c.hold_s:
            openness = 0.0
        elif t < total:
            openness = _min_jerk((t - c.close_s - c.hold_s) / c.open_s)
        else:
            openness = 1.0
            self._phase_t = None
            self._t = 0.0
            # Tuplaräpäytys: toinen räpäytys heti perään, sitten normaali väli.
            if not self._pending_double and self.rng.random() < c.double_blink_prob:
                self._pending_double = True
                self._next = 0.08
            else:
                self._pending_double = False
                self._next = self._draw_interval()
        return openness


def lid_angles(cfg: LidConfig, openness: float, pitch: float) -> tuple:
    up = cfg.closed + (cfg.upper_open + cfg.upper_follow_pitch * pitch - cfg.closed) * openness
    lo = cfg.closed + (cfg.lower_open + cfg.lower_follow_pitch * pitch - cfg.closed) * openness
    return up, min(lo, up)


class EyeController:
    """Yhdistää katseen ja räpäytykset yhdeksi tilaksi."""

    def __init__(self, cfg: EyeConfig, seed: int | None = None):
        self.cfg = cfg
        rng = random.Random(seed)
        self.gaze = GazeController(cfg.gaze, rng)
        self.blink = BlinkController(cfg.blink, rng)
        self.state = EyeState()

    def update(self, target: tuple | None, dt: float) -> EyeState:
        self.gaze.update(target, dt)
        openness = self.blink.update(dt, self.gaze.last_saccade_amp)
        up, lo = lid_angles(self.cfg.lids, openness, self.gaze.pitch)
        self.state = EyeState(
            yaw=self.gaze.yaw,
            pitch=self.gaze.pitch,
            upper_lid=up,
            lower_lid=lo,
            lid_open=openness,
            tracking=target is not None,
        )
        return self.state
