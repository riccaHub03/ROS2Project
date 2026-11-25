#!/usr/bin/env python3
"""Registra la RealSense (colore + profondita') in un file .db3 con anteprima dal vivo.

Dalla versione 2.59 librealsense registra nel formato rosbag2 (.db3, non piu' il vecchio .bag di ROS 1).
Va lanciato dove la camera trasmette (su un PC Windows: l'ambiente conda `realsense`). Il .db3 si apre col RealSense
Viewer, si esporta con scripts/bag_export.py e in ROS 2 si riproduce come se fosse la camera:
  ros2 run realsense2_camera realsense2_camera_node --ros-args -r __ns:=/ -r __node:=camera \
      -p rosbag_filename:=/percorso/file.db3 -p align_depth.enable:=true

  python scripts/realsense_record.py                   # anteprima: R avvia/mette in pausa, Q chiude
  python scripts/realsense_record.py --seconds 60      # conto alla rovescia, registra 60 s e si chiude
  python scripts/realsense_record.py --no-depth        # solo colore (basta per la calibrazione intrinseca)
  python scripts/realsense_record.py --color 1280x720@15 --depth 848x480@15

Senza --color/--depth sceglie profili adatti al collegamento: USB 3 -> colore 1280x720 e profondita' 848x480 a 15 fps;
USB 2 -> 640x480 a 15 fps o il profilo piu' vicino che la camera offre.

Si puo' lanciare da un terminale WSL col Python di Windows (che vede la camera), dentro la cartella del progetto:
  /mnt/c/Users/<utente>/miniconda3/envs/realsense/python.exe scripts/realsense_record.py --seconds 60
Il file si scrive prima sul disco locale (la cartella WSL vista da Windows e' lenta e poco adatta a SQLite) e a fine
registrazione si sposta in --out.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

WINDOW = "Registrazione RealSense"


def video_profiles(dev, stream, fmt) -> list[tuple[int, int, int]]:
    found = set()
    for sensor in dev.query_sensors():
        for p in sensor.get_stream_profiles():
            if p.stream_type() == stream and p.format() == fmt and p.is_video_stream_profile():
                v = p.as_video_stream_profile()
                found.add((v.width(), v.height(), p.fps()))
    return sorted(found)


def pick(profiles, width: int, fps: int):
    if not profiles:
        return None
    return min(profiles, key=lambda p: (abs(p[2] - fps), abs(p[0] - width)))


def parse_profile(text: str) -> tuple[int, int, int]:
    size, fps = text.lower().split("@")
    w, h = size.split("x")
    return int(w), int(h), int(fps)


def put(img, text, y, color=(255, 255, 255), scale=0.7):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="file .db3 (default data/recordings/rs_<data>_<ora>.db3)")
    ap.add_argument("--seconds", type=float, default=0, help="registra per N secondi e chiude (0 = manuale con R)")
    ap.add_argument("--countdown", type=float, default=5, help="con --seconds: attesa prima di iniziare")
    ap.add_argument("--no-depth", action="store_true", help="solo colore")
    ap.add_argument("--color", help="profilo colore LARGHEZZAxALTEZZA@FPS")
    ap.add_argument("--depth", help="profilo profondita' LARGHEZZAxALTEZZA@FPS")
    ap.add_argument("--no-preview", action="store_true", help="nessuna finestra (richiede --seconds)")
    args = ap.parse_args()
    if args.no_preview and not args.seconds:
        sys.exit("--no-preview richiede --seconds")

    devs = rs.context().query_devices()
    if not len(devs):
        sys.exit("Nessuna RealSense trovata: lancia prima scripts/realsense_probe.py")
    dev = devs[0]
    serial = dev.get_info(rs.camera_info.serial_number)
    usb = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "?"
    usb3 = usb.startswith("3")

    color_avail = video_profiles(dev, rs.stream.color, rs.format.bgr8)
    depth_avail = video_profiles(dev, rs.stream.depth, rs.format.z16)
    c = parse_profile(args.color) if args.color else pick(color_avail, 1280 if usb3 else 640, 15)
    if c is None:
        sys.exit("La camera non offre profili colore con questo collegamento.")
    d = None
    if not args.no_depth:
        if args.depth:
            d = parse_profile(args.depth)
        else:
            same_fps = [p for p in depth_avail if p[2] == c[2]]
            d = pick(same_fps or depth_avail, 848 if usb3 else 640, c[2])
        if d is None:
            print("La camera non offre profondita' con questo collegamento: registro solo il colore.")

    out = Path(args.out) if args.out else Path("data/recordings") / time.strftime("rs_%Y%m%d_%H%M%S.db3")
    if out.suffix != ".db3":  # librealsense 2.59 rifiuta altre estensioni
        out = out.with_suffix(".db3")
        print(f"librealsense registra solo in formato .db3: salvo in {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.gettempdir()) / out.name  # disco locale: veloce, poi spostato in `out`

    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, c[0], c[1], rs.format.bgr8, c[2])
    if d:
        cfg.enable_stream(rs.stream.depth, d[0], d[1], rs.format.z16, d[2])
    cfg.enable_record_to_file(str(tmp))
    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    recorder = profile.get_device().as_recorder()
    recorder.pause()  # si registra solo dopo il conto alla rovescia o col tasto R
    desc = f"USB {usb}, colore {c[0]}x{c[1]}@{c[2]}" + (f", profondita' {d[0]}x{d[1]}@{d[2]}" if d else ", senza profondita'")
    print(f"Camera {serial}: {desc}\nFile: {out}")

    state = {"on": False, "total": 0.0, "since": 0.0}

    def set_recording(on: bool) -> None:
        if on and not state["on"]:
            recorder.resume()
            state.update(on=True, since=time.time())
        elif not on and state["on"]:
            recorder.pause()
            state.update(on=False, total=state["total"] + time.time() - state["since"])

    def recorded() -> float:
        return state["total"] + (time.time() - state["since"] if state["on"] else 0.0)

    colorizer = rs.colorizer()
    start_at = time.time() + args.countdown if args.seconds else None
    frames_seen, last_frame = 0, time.time()
    view = np.zeros((c[1], c[0] * (2 if d else 1), 3), np.uint8)
    try:
        while True:
            ok, frames = pipe.try_wait_for_frames(1000)
            now = time.time()
            if ok:
                frames_seen += 1
                last_frame = now
            if start_at is not None and now >= start_at and not state["on"] and state["total"] == 0.0:
                set_recording(True)
                print("Registrazione avviata.")
            if args.seconds and state["on"] and recorded() >= args.seconds:
                break
            if args.no_preview:
                if now - last_frame > 10:
                    print("Nessun frame da 10 s: interrompo.")
                    break
                continue

            if ok and frames.get_color_frame():
                img = np.asanyarray(frames.get_color_frame().get_data())
                depth = frames.get_depth_frame() if d else None
                if depth:
                    dimg = cv2.cvtColor(np.asanyarray(colorizer.colorize(depth).get_data()), cv2.COLOR_RGB2BGR)
                    dimg = cv2.resize(dimg, (round(dimg.shape[1] * img.shape[0] / dimg.shape[0]), img.shape[0]))
                    view = np.hstack([img, dimg])
                else:
                    view = img.copy()
            shown = view.copy()
            if now - last_frame > 2:
                put(shown, "NESSUN FRAME dalla camera (vedi realsense_probe.py)", 30, (60, 60, 255))
            elif state["on"]:
                secs = recorded()
                put(shown, f"REC {int(secs // 60):02d}:{int(secs % 60):02d}" + ("" if args.seconds else "  (R pausa)"), 30, (60, 60, 255))
            elif start_at is not None and now < start_at:
                put(shown, f"Si registra tra {math.ceil(start_at - now)} s", 30, (60, 220, 255))
            else:
                put(shown, "PAUSA - R registra, Q chiude", 30, (60, 220, 255))
            put(shown, desc, shown.shape[0] - 14, scale=0.5)
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r") and not args.seconds:
                set_recording(not state["on"])
    finally:
        set_recording(False)
        pipe.stop()
        cv2.destroyAllWindows()
    # il file resta aperto finche' esistono riferimenti al dispositivo di registrazione: rilasciarli prima di spostarlo
    del recorder, profile, pipe
    gc.collect()

    total = state["total"]
    if total == 0.0:
        tmp.unlink(missing_ok=True)
        print("Non e' stato registrato nulla: nessun file salvato.")
        return 1 if frames_seen == 0 else 0
    print(f"Sposto la registrazione in {out.as_posix()} ...")
    for attempt in range(10):
        try:
            shutil.move(str(tmp), str(out))
            break
        except PermissionError:  # su Windows il file puo' restare bloccato per un attimo dopo la chiusura
            if attempt == 9:
                print(f"Non riesco a spostare la registrazione: e' rimasta in {tmp}")
                out = tmp
                break
            time.sleep(0.5)
    meta = {
        "file": out.as_posix(),
        "serial": serial,
        "usb_type": usb,
        "color_profile": list(c),
        "depth_profile": list(d) if d else None,
        "recorded_s": round(total, 1),
        "frames_seen": frames_seen,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    size_mb = out.stat().st_size / 1e6 if out.exists() else 0
    print(f"Registrati {total:.1f} s in {out.as_posix()} ({size_mb:.0f} MB).")
    print(f"Esporta con: python3 scripts/bag_export.py {out.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
