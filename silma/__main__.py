"""Käynnistys: python -m silma [--camera N | --video tiedosto | --demo] [--record out.mp4]

Ilman --camera-valintaa ohjelma listaa kamerat ja kysyy, mitä käytetään.

Näppäimet simulaattori-ikkunassa:
  a/d, w/s  kierrä katselukulmaa     r  palauta näkymä
  b         räpäytä                   q / Esc  lopeta
"""

import argparse
import time
from math import cos, radians, sin

import cv2
import numpy as np

from .behavior import EyeController
from .config import EyeConfig
from .face_tracker import FaceDetector, face_to_target, select_face
from .renderer import EyeRenderer


class SyntheticTarget:
    """Testikohde ilman kameraa: kasvot liikkuvat ja välillä katoavat."""

    def __init__(self):
        self.t = 0.0

    def update(self, dt):
        self.t += dt
        t = self.t
        if 6.0 < t % 14.0 < 9.0:
            return None  # kasvot poissa näkyvistä -> silmä vilkuilee
        yaw = radians(25) * sin(t * 0.7) + (radians(10) if int(t / 3) % 2 else 0)
        pitch = radians(10) * sin(t * 1.3) * cos(t * 0.4)
        return yaw, pitch


def _overlay(img, lines):
    for i, text in enumerate(lines):
        cv2.putText(img, text, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Kasvoja seuraava silmä – simulaattori")
    ap.add_argument("--camera", type=int, help="kameran numero (ohittaa valintavalikon)")
    ap.add_argument("--video", help="videotiedosto kameran sijaan")
    ap.add_argument("--demo", action="store_true", help="synteettinen kohde, ei kameraa")
    ap.add_argument("--mirrored", action="store_true", help="kamerakuva on peilattu")
    ap.add_argument("--headless", action="store_true", help="ei ikkunaa (käytä --record)")
    ap.add_argument("--record", help="tallenna video (mp4)")
    ap.add_argument("--seconds", type=float, default=0, help="lopeta N sekunnin jälkeen")
    ap.add_argument("--size", type=int, default=400, help="simulaattorikuvan koko")
    ap.add_argument("--servos", action="store_true", help="ohjaa myös PCA9685-servoja")
    args = ap.parse_args()

    cfg = EyeConfig()
    if args.camera is not None:
        cfg.camera.index = args.camera
    elif not (args.demo or args.video):
        from .camera_select import choose_camera

        cfg.camera.index = choose_camera()
    cfg.camera.mirrored = args.mirrored
    controller = EyeController(cfg)
    renderer = EyeRenderer(args.size)
    servo_out = None
    if args.servos:
        from .servos import ServoOutput

        servo_out = ServoOutput()

    cap = detector = synthetic = None
    if args.demo:
        synthetic = SyntheticTarget()
    else:
        cap = cv2.VideoCapture(args.video if args.video else cfg.camera.index)
        if not args.video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
        if not cap.isOpened():
            raise SystemExit("Kameraa/videota ei saatu auki. Kokeile --demo.")
        detector = FaceDetector(cfg.detector)

    # Videolla ja demossa käytetään tasaista aika-askelta, jotta tallenne
    # vastaa reaaliaikaa riippumatta laskentanopeudesta.
    fixed_dt = None
    if args.demo:
        fixed_dt = 1 / 30
    elif args.video:
        fixed_dt = 1 / (cap.get(cv2.CAP_PROP_FPS) or 30)

    writer = None
    view_yaw = view_pitch = 0.0
    face = None
    sim_t = 0.0
    last = time.monotonic()
    fps = 0.0

    while True:
        now = time.monotonic()
        dt = fixed_dt or min(now - last, 0.1)
        if not fixed_dt:
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
        last = now
        sim_t += dt

        cam_view = None
        if synthetic:
            target = synthetic.update(dt)
        else:
            ok, frame = cap.read()
            if not ok:
                break
            face = select_face(detector.detect(frame), face)
            target = None
            if face is not None:
                h, w = frame.shape[:2]
                tgt = face_to_target(face, (w, h), cfg.camera, cfg.detector.interocular_m)
                target = (tgt.yaw, tgt.pitch)
                x, y, bw, bh = (int(v) for v in face.box)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.circle(frame, tuple(int(v) for v in face.eyes_mid), 4, (0, 0, 255), -1)
                _overlay(frame, [f"etaisyys ~{tgt.position_m[2]:.2f} m"])
            cam_view = cv2.resize(frame, (int(frame.shape[1] * args.size / frame.shape[0]), args.size))

        state = controller.update(target, dt)
        if servo_out:
            servo_out.apply(state)
        eye = renderer.render(state)
        _overlay(
            eye,
            [
                f"yaw {np.degrees(state.yaw):+5.1f}  pitch {np.degrees(state.pitch):+5.1f}",
                f"luomi {state.lid_open:.2f}  {'seuraa' if state.tracking else 'etsii'}",
            ]
            + ([f"{fps:4.1f} fps"] if not fixed_dt else []),
        )
        out = np.hstack([cam_view, eye]) if cam_view is not None else eye

        if args.record:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.record, fourcc, 1 / (fixed_dt or 1 / 30), (out.shape[1], out.shape[0]))
            writer.write(out)

        if not args.headless:
            cv2.imshow("silma", out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            step = radians(10)
            moves = {ord("a"): (-step, 0), ord("d"): (step, 0), ord("w"): (0, step), ord("s"): (0, -step)}
            if key in moves:
                view_yaw += moves[key][0]
                view_pitch = max(-1.2, min(1.2, view_pitch + moves[key][1]))
                renderer.set_view(view_yaw, view_pitch)
            elif key == ord("r"):
                view_yaw = view_pitch = 0.0
                renderer.set_view(0, 0)
            elif key == ord("b"):
                controller.blink.trigger()

        if args.seconds and sim_t >= args.seconds:
            break

    if writer:
        writer.release()
    if cap:
        cap.release()
    cv2.destroyAllWindows() if not args.headless else None


if __name__ == "__main__":
    main()
