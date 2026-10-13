"""Ihmishahmojen tunnistus NanoDet-Plus-mallilla (COCO-luokka 0 = henkilö).

Esi- ja jälkikäsittely seuraa OpenCV Zoon viitetoteutusta
(models/object_detection_nanodet, Apache 2.0).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import PersonConfig

INPUT = 416
REG_MAX = 7
MEAN = np.array([103.53, 116.28, 123.675], np.float32)
STD = np.array([57.375, 57.12, 58.395], np.float32)
PERSON_CLASS = 0


@dataclass
class Person:
    box: tuple  # (x, y, w, h) täysikokoisen kuvan pikseleinä
    score: float


class PersonDetector:
    def __init__(self, cfg: PersonConfig):
        if not cfg.model_path.exists():
            raise FileNotFoundError(f"NanoDet-mallia ei löydy: {cfg.model_path}")
        self.cfg = cfg
        self._net = cv2.dnn.readNet(str(cfg.model_path))
        self._anchors = {}
        self._project = np.arange(REG_MAX + 1, dtype=np.float32)

    def detect(self, frame_bgr) -> list:
        h, w = frame_bgr.shape[:2]
        # Letterbox: skaalaa pidempi sivu 416:een, täytä loput mustalla.
        scale = INPUT / max(h, w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        left, top = (INPUT - nw) // 2, (INPUT - nh) // 2
        canvas = np.zeros((INPUT, INPUT, 3), np.uint8)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        canvas[top : top + nh, left : left + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage((canvas.astype(np.float32) - MEAN) / STD)
        self._net.setInput(blob)
        outs = self._net.forward(self._net.getUnconnectedOutLayersNames())

        boxes, scores = [], []
        for cls, reg, anchors, stride in self._levels(outs):
            person = cls[:, PERSON_CLASS]
            keep = person >= self.cfg.score_threshold
            if not np.any(keep):
                continue
            reg = reg.reshape(-1, 4, REG_MAX + 1)[keep]
            e = np.exp(reg - reg.max(axis=-1, keepdims=True))
            dist = (e / e.sum(axis=-1, keepdims=True)) @ self._project * stride
            a = anchors[keep]
            x1 = a[:, 0] - dist[:, 0]
            y1 = a[:, 1] - dist[:, 1]
            x2 = a[:, 0] + dist[:, 2]
            y2 = a[:, 1] + dist[:, 3]
            boxes.append(np.column_stack((x1, y1, x2 - x1, y2 - y1)))
            scores.append(person[keep])
        if not boxes:
            return []
        boxes = np.concatenate(boxes)
        scores = np.concatenate(scores)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.cfg.score_threshold, 0.6)

        people = []
        for i in np.array(idx).ravel():
            x, y, bw, bh = boxes[i]
            # Takaisin alkuperäisen kuvan koordinaatteihin, rajattuna kuvaan.
            x0 = np.clip((x - left) / scale, 0, w)
            y0 = np.clip((y - top) / scale, 0, h)
            x1 = np.clip((x + bw - left) / scale, 0, w)
            y1 = np.clip((y + bh - top) / scale, 0, h)
            if x1 - x0 > 1 and y1 - y0 > 1:
                people.append(Person(box=(x0, y0, x1 - x0, y1 - y0), score=float(scores[i])))
        return people

    def _levels(self, outs):
        """Parita luokka- (80 saraketta) ja laatikkoulostulot (32) tasoittain.

        Ulostulojen järjestys vaihtelee OpenCV-versioiden välillä, joten
        tasot tunnistetaan muodosta: n x n -ruudukko -> stride = 416 / n.
        """
        cls_by_rows, reg_by_rows = {}, {}
        for o in outs:
            o = o.reshape(-1, o.shape[-1])
            (cls_by_rows if o.shape[1] == 80 else reg_by_rows)[o.shape[0]] = o
        for rows, cls in cls_by_rows.items():
            n = int(round(rows**0.5))
            stride = INPUT // n
            if rows not in self._anchors:
                xv, yv = np.meshgrid(np.arange(n) * stride, np.arange(n) * stride)
                self._anchors[rows] = np.column_stack(
                    (xv.ravel() + 0.5 * (stride - 1), yv.ravel() + 0.5 * (stride - 1))
                )
            yield cls, reg_by_rows[rows], self._anchors[rows], stride
