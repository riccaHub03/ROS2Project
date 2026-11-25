#!/usr/bin/env python3
"""Diagnosi di una camera RealSense: la vede? Con che collegamento USB? Arrivano davvero i frame?

Va lanciato dove la camera e' collegata e ha i driver (su un PC Windows: l'ambiente conda `realsense`).
Prova, per qualche secondo ciascuno, solo colore, solo profondita' e i due insieme, scegliendo profili che il
collegamento attuale supporta, e dice quanti frame al secondo arrivano.

  python scripts/realsense_probe.py             # diagnosi
  python scripts/realsense_probe.py --reset     # prima riavvia la camera (reset hardware), poi la diagnosi

Esce con 0 se il colore trasmette, 1 se la camera non c'e', 2 se c'e' ma non trasmette.
"""
from __future__ import annotations

import argparse
import sys
import time

import pyrealsense2 as rs

INFO_KEYS = ("name", "serial_number", "firmware_version", "usb_type_descriptor", "physical_port")


def describe(dev) -> dict[str, str]:
    info = {}
    for key in INFO_KEYS:
        field = getattr(rs.camera_info, key)
        info[key] = dev.get_info(field) if dev.supports(field) else "-"
    return info


def video_profiles(dev, stream, fmt) -> list[tuple[int, int, int]]:
    """Profili (larghezza, altezza, fps) che la camera offre per quel flusso, con il collegamento attuale."""
    found = set()
    for sensor in dev.query_sensors():
        for p in sensor.get_stream_profiles():
            if p.stream_type() == stream and p.format() == fmt and p.is_video_stream_profile():
                v = p.as_video_stream_profile()
                found.add((v.width(), v.height(), p.fps()))
    return sorted(found)


def pick(profiles, width: int, fps: int):
    """Il profilo piu' vicino a (larghezza, fps) richiesti, con priorita' agli fps."""
    if not profiles:
        return None
    return min(profiles, key=lambda p: (abs(p[2] - fps), abs(p[0] - width)))


def wait_for_device(timeout_s: float):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        devs = rs.context().query_devices()
        if len(devs):
            return devs[0]
        time.sleep(0.5)
    return None


def run_test(serial: str, streams, seconds: float):
    """Avvia i flussi e restituisce gli fps ricevuti per ciascuno, oppure il messaggio d'errore."""
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(serial)
    for stream, fmt, (w, h, fps) in streams:
        cfg.enable_stream(stream, w, h, fmt, fps)
    counts = {stream: 0 for stream, _, _ in streams}
    try:
        pipe.start(cfg)
    except RuntimeError as exc:
        return None, f"avvio rifiutato: {exc}"
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            ok, frames = pipe.try_wait_for_frames(1000)
            if not ok:
                continue
            for stream in counts:
                if frames.first_or_default(stream):
                    counts[stream] += 1
    finally:
        pipe.stop()
    return {stream: n / seconds for stream, n in counts.items()}, None


def fmt_profile(p) -> str:
    return f"{p[0]}x{p[1]}@{p[2]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=4.0, help="durata di ogni prova")
    ap.add_argument("--reset", action="store_true", help="reset hardware della camera prima delle prove")
    args = ap.parse_args()

    dev = wait_for_device(2.0)
    if dev is None:
        print("Nessuna RealSense trovata. Controlla che sia collegata e che nessun altro programma la stia usando")
        print("(RealSense Viewer, usbipd attach verso WSL).")
        return 1
    if args.reset:
        print("Reset hardware della camera...")
        dev.hardware_reset()
        time.sleep(3)
        dev = wait_for_device(20.0)
        if dev is None:
            print("Dopo il reset la camera non e' ricomparsa: scollegala e ricollegala.")
            return 1
        time.sleep(2)

    info = describe(dev)
    for key in INFO_KEYS:
        print(f"  {key:20s} {info[key]}")
    usb = info["usb_type_descriptor"]
    if not usb.startswith("3"):
        print(f"\nATTENZIONE: collegamento USB {usb}. Con USB 2 la profondita' e' limitata o assente;")
        print("serve una porta USB 3 e un cavo USB 3 (il Viewer deve mostrare 3.2).")

    color = video_profiles(dev, rs.stream.color, rs.format.bgr8)
    depth = video_profiles(dev, rs.stream.depth, rs.format.z16)
    print(f"\nProfili offerti: colore {len(color)}, profondita' {len(depth)}")

    tests = []
    c, d = pick(color, 640, 15), pick(depth, 640, 15)
    if c:
        tests.append(("solo colore", [(rs.stream.color, rs.format.bgr8, c)]))
    if d:
        tests.append(("solo profondita'", [(rs.stream.depth, rs.format.z16, d)]))
    common = sorted({p[2] for p in color} & {p[2] for p in depth}, key=lambda f: abs(f - 15))
    if common:
        fps = common[0]
        c2 = pick([p for p in color if p[2] == fps], 640, fps)
        d2 = pick([p for p in depth if p[2] == fps], 640, fps)
        tests.append(("colore + profondita'", [(rs.stream.color, rs.format.bgr8, c2), (rs.stream.depth, rs.format.z16, d2)]))

    results = {}
    serial = info["serial_number"]
    for name, streams in tests:
        label = " + ".join(fmt_profile(p) for _, _, p in streams)
        res, err = run_test(serial, streams, args.seconds)
        if err:
            print(f"  {name:22s} {label:28s} {err}")
            results[name] = 0.0
            continue
        rates = ", ".join(f"{str(s).split('.')[-1]} {r:.1f} fps" for s, r in res.items())
        print(f"  {name:22s} {label:28s} {rates}")
        results[name] = min(res.values())
        time.sleep(0.5)

    color_ok = results.get("solo colore", 0.0) > 1.0
    depth_ok = results.get("solo profondita'", 0.0) > 1.0
    print()
    if color_ok and depth_ok:
        print("ESITO: la camera trasmette colore e profondita'.")
        return 0
    if color_ok:
        print("ESITO: il colore trasmette, la profondita' no. Tipico del collegamento USB 2: cambia porta o cavo.")
        return 0
    print("ESITO: la camera e' vista ma NON trasmette. Non e' un problema di installazione: e' il collegamento")
    print("(cavo o porta USB) oppure un filtro USB di Windows (USBPcap, VBoxUSBMon di usbipd). Prova --reset,")
    print("poi un altro cavo/porta.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
