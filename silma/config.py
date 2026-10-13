"""Kaikki säädettävät parametrit yhdessä paikassa.

Koordinaatisto (lelun oma, katsottuna lelun edestä):
  +x = katsojan oikealle (= lelun vasemmalle)
  +y = ylös
  +z = lelusta ulospäin, kohti katsojaa
Silmämunan keskipiste on origossa. yaw > 0 kääntää katseen +x-suuntaan,
pitch > 0 ylöspäin. Kulmat ovat radiaaneja ellei toisin mainita.
"""

from dataclasses import dataclass, field
from math import radians
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "mallit"


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    # Vaakasuuntainen näkökenttä. Tyypillinen web-kamera ~60-70°,
    # Raspberry Pi Camera Module 3 ~66°, laajakulmaversio ~102°.
    hfov_deg: float = 66.0
    # Kameran sijainti silmämunan keskipisteeseen nähden (metreinä).
    # Kamera on silmän yläpuolella -> y > 0.
    offset_m: tuple = (0.0, 0.05, 0.0)
    # True, jos kamerakuva on peilattu (esim. jotkin web-kamerat/ajurit).
    mirrored: bool = False


@dataclass
class DetectorConfig:
    model_path: Path = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
    # Tunnistus ajetaan pienennetyllä kuvalla: nopeampi, riittää lähikasvoille.
    detect_width: int = 320
    score_threshold: float = 0.7
    # Keskimääräinen silmien välinen etäisyys etäisyysarviota varten.
    interocular_m: float = 0.063


@dataclass
class GazeConfig:
    max_yaw: float = radians(35)
    max_pitch: float = radians(25)
    # Saccadi (nopea hyppy) laukeaa, kun kohde on näin kaukana nykyisestä
    # fiksaatiosta. Pienemmät erot seurataan hitaasti (smooth pursuit).
    saccade_threshold: float = radians(3.0)
    # Ihmisen "main sequence": kesto ≈ 20 ms + 2.5 ms/aste.
    saccade_base_s: float = 0.020
    saccade_per_deg_s: float = 0.0025
    pursuit_rate: float = 4.0  # 1/s, eksponentiaalinen lähestyminen
    # Kuinka kauan kasvojen katoamisen jälkeen katsotaan viimeiseen kohtaan
    # ennen kuin aletaan vilkuilla ympärille.
    lost_hold_s: float = 1.5
    idle_glance_interval_s: tuple = (0.8, 3.0)
    idle_range: float = radians(20)
    # Pienet mikrosakkadit fiksaation aikana tekevät silmästä elävämmän.
    microsaccade_interval_s: tuple = (0.6, 2.0)
    microsaccade_amp: float = radians(0.6)


@dataclass
class BlinkConfig:
    # Ihminen räpäyttää ~15-20 krt/min levossa. Väli arvotaan
    # log-normaalijakaumasta, jolloin välit vaihtelevat luonnollisesti.
    mean_interval_s: float = 3.5
    interval_sigma: float = 0.5
    min_interval_s: float = 0.8
    double_blink_prob: float = 0.12
    # Suuri katseen siirto laukaisee usein räpäytyksen (gaze-evoked blink).
    saccade_blink_prob: float = 0.35
    saccade_blink_min_amp: float = radians(15)
    close_s: float = 0.075
    hold_s: float = 0.030
    open_s: float = 0.160


@dataclass
class LidConfig:
    # Luomien reunojen kulmat (elevaatio silmän keskipisteestä, x-akselin
    # ympäri). Auki: yläluomi ylhäällä, alaluomi alhaalla.
    upper_open: float = radians(38)
    lower_open: float = radians(-42)
    # Kiinni: reunat kohtaavat hieman keskilinjan alapuolella.
    closed: float = radians(-8)
    # Yläluomi seuraa katseen pystysuuntaa (alas katsottaessa luomi laskee).
    upper_follow_pitch: float = 0.6
    lower_follow_pitch: float = 0.3


@dataclass
class EyeConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    blink: BlinkConfig = field(default_factory=BlinkConfig)
    lids: LidConfig = field(default_factory=LidConfig)
