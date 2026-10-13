"""Kameran valinta komentorivillä ennen simulaattorin käynnistystä."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import cv2

LAST_CHOICE_FILE = Path(__file__).resolve().parent.parent / ".viimeisin_kamera"


def probe_cameras(max_index: int = 8) -> list:
    """Kokeile kameraindeksit järjestyksessä. Palauttaa [(indeksi, leveys, korkeus)]."""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        pass
    found = []
    misses = 0
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened() and cap.read()[0]
        if ok:
            found.append((i, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            misses = 0
        else:
            misses += 1
        cap.release()
        # Indeksit ovat yleensä peräkkäisiä: kaksi tyhjää peräkkäin -> lopeta.
        if misses >= 2:
            break
    return found


def system_camera_names() -> tuple:
    """Käyttöjärjestelmän kameranimet ja tieto, vastaako järjestys indeksejä.

    Linuxissa indeksi N on /dev/videoN, joten nimet ovat luotettavat.
    macOS:llä nimet saadaan system_profilerista, mutta niiden järjestys ei
    välttämättä vastaa OpenCV:n indeksejä.
    """
    system = platform.system()
    if system == "Linux":
        names = {}
        for p in Path("/sys/class/video4linux").glob("video*"):
            try:
                names[int(p.name[5:])] = (p / "name").read_text().strip()
            except (ValueError, OSError):
                pass
        return names, True
    if system == "Darwin":
        try:
            out = subprocess.run(
                ["system_profiler", "SPCameraDataType", "-json"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            return parse_macos_camera_names(out), False
        except (OSError, subprocess.SubprocessError):
            pass
    return {}, False


def parse_macos_camera_names(json_text: str) -> dict:
    try:
        cams = json.loads(json_text).get("SPCameraDataType", [])
    except ValueError:
        return {}
    return {i: c.get("_name", "?") for i, c in enumerate(cams)}


def load_last_choice() -> int | None:
    try:
        return int(LAST_CHOICE_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def save_last_choice(index: int) -> None:
    try:
        LAST_CHOICE_FILE.write_text(str(index))
    except OSError:
        pass


def parse_choice(text: str, available: list, default: int) -> int | None:
    """Tulkitse käyttäjän syöte. Tyhjä = oletus. None = virheellinen."""
    text = text.strip()
    if not text:
        return default
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value in available else None


def choose_camera() -> int:
    """Listaa kamerat ja kysy käyttäjältä. Palauttaa valitun indeksin."""
    print("Etsitään kameroita...", flush=True)
    cams = probe_cameras()
    if not cams:
        raise SystemExit(
            "Yhtään kameraa ei löytynyt. Tarkista, että Terminalilla on lupa "
            "käyttää kameraa (macOS: Järjestelmäasetukset → Tietosuoja ja "
            "suojaus → Kamera), tai kokeile --demo."
        )
    available = [c[0] for c in cams]
    last = load_last_choice()
    default = last if last in available else available[0]

    names, names_match = system_camera_names()
    print("\nLöydetyt kamerat:")
    for i, w, h in cams:
        name = f"  {names[i]}" if names_match and i in names else ""
        mark = "  <- oletus" if i == default else ""
        print(f"  [{i}] {w}x{h}{name}{mark}")
    if names and not names_match:
        print("\nJärjestelmän kamerat (järjestys ei välttämättä vastaa numeroita):")
        for n in names.values():
            print(f"  - {n}")

    if len(cams) == 1 or not sys.stdin.isatty():
        choice = default
    else:
        while True:
            text = input(f"\nValitse kamera numerolla (Enter = {default}): ")
            choice = parse_choice(text, available, default)
            if choice is not None:
                break
            print(f"Anna jokin näistä: {', '.join(map(str, available))}")
    print(f"Käytetään kameraa {choice}. Väärä kamera? Lopeta q:lla ja valitse uudelleen.\n")
    save_last_choice(choice)
    return choice
