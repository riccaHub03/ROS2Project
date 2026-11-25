#!/usr/bin/env python3
"""Esporta una registrazione RealSense in file usabili dal progetto.

Legge il formato .db3 (rosbag2) che librealsense 2.59 e il RealSense Viewer producono, e i .bag delle versioni
precedenti. Non serve la camera. Dipendenze: pyrealsense2, numpy, opencv-python (vedi requirements-calibration.txt).

  python3 scripts/bag_export.py data/recordings/prova.db3
  python3 scripts/bag_export.py data/recordings/prova.db3 --every 10 --out data/calibration/prova

Produce in --out (default data/calibration/<nome della registrazione>):
  color.mp4              video a colori, per scripts/calibrate_camera.py --video color.mp4
  intrinsics_factory.json  intrinseche di fabbrica del colore (punto di partenza, NON sostituiscono la calibrazione)
  frames/color_000123.png  un frame ogni --every, con la profondita' allineata al colore:
  frames/depth_000123.png  16 bit, millimetri (0 = nessuna misura)
  info.json              risoluzione, fps, scala della profondita', numero di frame
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--out", help="cartella di destinazione")
    ap.add_argument("--every", type=int, default=10, help="salva un frame PNG (colore+profondita') ogni N")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = tutto il file")
    args = ap.parse_args()

    bag = Path(args.bag)
    if not bag.is_file():
        sys.exit(f"File non trovato: {bag}")
    out = Path(args.out) if args.out else Path("data/calibration") / bag.stem
    (out / "frames").mkdir(parents=True, exist_ok=True)

    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, str(bag), repeat_playback=False)
    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    profile.get_device().as_playback().set_real_time(False)
    align = rs.align(rs.stream.color)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    width, height, fps = intr.width, intr.height, color_profile.fps()
    depth_scale = None
    for s in profile.get_device().query_sensors():
        if s.is_depth_sensor():
            depth_scale = s.as_depth_sensor().get_depth_scale()
    has_depth = True
    try:
        profile.get_stream(rs.stream.depth)
    except RuntimeError:
        has_depth = False

    writer = cv2.VideoWriter(str(out / "color.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    (out / "intrinsics_factory.json").write_text(
        json.dumps(
            {
                "image_width_px": width,
                "image_height_px": height,
                "fx_px": intr.fx,
                "fy_px": intr.fy,
                "cx_px": intr.ppx,
                "cy_px": intr.ppy,
                "distortion_model": str(intr.model),
                "distortion_coeffs": list(intr.coeffs),
                "note": "intrinseche di fabbrica lette dal bag; per la presa reale usare la calibrazione misurata",
            },
            indent=2,
        )
    )

    n = saved = 0
    last_ts = -1.0
    try:
        while True:
            try:
                frames = pipe.wait_for_frames(2000)
            except RuntimeError:
                break  # fine del file
            ts = frames.get_timestamp()
            if ts < last_ts:  # il bag e' ricominciato da capo
                break
            last_ts = ts
            if has_depth:
                frames = align.process(frames)
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            if color.get_profile().format() == rs.format.rgb8:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(img)
            if has_depth and n % args.every == 0:
                depth = frames.get_depth_frame()
                if depth:
                    cv2.imwrite(str(out / "frames" / f"color_{n:06d}.png"), img)
                    cv2.imwrite(str(out / "frames" / f"depth_{n:06d}.png"), np.asanyarray(depth.get_data()))
                    saved += 1
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        writer.release()
        pipe.stop()

    (out / "info.json").write_text(
        json.dumps(
            {
                "source_bag": str(bag),
                "width": width,
                "height": height,
                "fps": fps,
                "frames": n,
                "frames_with_depth_saved": saved,
                "depth_scale_m_per_unit": depth_scale,
                "has_depth": has_depth,
            },
            indent=2,
        )
    )
    print(f"{n} frame letti -> {out}/color.mp4; {saved} coppie colore+profondita' in {out}/frames")
    if n == 0:
        print("Nessun frame: il bag e' vuoto o non contiene il flusso colore.")
        return 1
    print(f"Calibrazione: python3 scripts/calibrate_camera.py --video {out}/color.mp4 --pattern 9x6 --square-mm 25")
    return 0


if __name__ == "__main__":
    sys.exit(main())
