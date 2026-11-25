#!/usr/bin/env python3
"""Calibrazione intrinseca della camera con anteprima dal vivo (OpenCV + scacchiera).

Dipendenze: numpy e opencv-python (con GUI, NON la variante headless); pyrealsense2 per --realsense
(vedi requirements-calibration.txt).

  python scripts/calibrate_camera.py --realsense --size 640x480 --fps 15   # RealSense dal vivo (anche su Windows)
  python3 scripts/calibrate_camera.py --device /dev/video4 --size 1280x720 # camera V4L2 su Linux
  python3 scripts/calibrate_camera.py --video calib_video.mp4              # da un video gia' registrato
  python3 scripts/calibrate_camera.py --video data/calibration/NOME/color.mp4  # colore esportato da un .db3

`--pattern` sono gli angoli INTERNI della scacchiera (colonne x righe): una scacchiera di 10x7 quadrati ha 9x6 angoli.

Tasti nella finestra:  SPAZIO cattura la posa   U annulla l'ultima   C calcola la calibrazione
                       D mostra/nasconde l'immagine corretta dalla distorsione   V avvia/ferma la registrazione
                       Q o Esc esce
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)


def parse_pair(text: str, sep: str) -> tuple[int, int]:
    a, b = text.lower().split(sep)
    return int(a), int(b)


class OpenCVSource:
    """Video da file, oppure camera letta da OpenCV (V4L2 su Linux, backend predefinito altrove)."""

    factory = None

    def __init__(self, args):
        if args.video:
            self.cap = cv2.VideoCapture(args.video)
        else:
            dev = int(args.device) if str(args.device).isdigit() else args.device
            backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(dev, backend)
            w, h = parse_pair(args.size, "x")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv2.CAP_PROP_FPS, args.fps)
        if not self.cap.isOpened():
            sys.exit(f"Impossibile aprire la sorgente: {args.video or args.device}")
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class RealSenseSource:
    """Flusso colore di una RealSense tramite pyrealsense2. Se il profilo chiesto non c'e' (per esempio con USB 2)
    usa quello con la risoluzione piu' vicina: le intrinseche valgono solo per la risoluzione calibrata."""

    def __init__(self, args):
        import pyrealsense2 as rs

        devs = rs.context().query_devices()
        if not len(devs):
            sys.exit("Nessuna RealSense trovata (diagnosi: scripts/realsense_probe.py).")
        w, h = parse_pair(args.size, "x")
        fps = args.fps
        avail = set()
        for sensor in devs[0].query_sensors():
            for p in sensor.get_stream_profiles():
                if p.stream_type() == rs.stream.color and p.format() == rs.format.bgr8 and p.is_video_stream_profile():
                    v = p.as_video_stream_profile()
                    avail.add((v.width(), v.height(), p.fps()))
        if avail and (w, h, fps) not in avail:
            w, h, fps = min(avail, key=lambda p: (abs(p[0] - w) + abs(p[1] - h), abs(p[2] - fps)))
            print(f"Profilo {args.size}@{args.fps} non disponibile con questo collegamento: uso {w}x{h}@{fps}")
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        try:
            profile = self.pipe.start(cfg)
        except RuntimeError as exc:
            sys.exit(f"RealSense: impossibile avviare il colore {w}x{h}@{fps}: {exc}")
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.size = (stream.width(), stream.height())
        i = stream.get_intrinsics()
        self.factory = {"fx_px": i.fx, "fy_px": i.fy, "cx_px": i.ppx, "cy_px": i.ppy, "coeffs": list(i.coeffs)}

    def read(self):
        ok, frames = self.pipe.try_wait_for_frames(1000)
        color = frames.get_color_frame() if ok else None
        if not color:
            return False, None
        return True, np.asanyarray(color.get_data()).copy()

    def release(self):
        self.pipe.stop()


def find_corners(gray: np.ndarray, pattern: tuple[int, int]):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calibrate(obj_pts, img_pts, size):
    rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    errors = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, k, dist)
        errors.append(float(np.sqrt(np.mean(np.sum((i - proj) ** 2, axis=2)))))
    return rms, k, dist, errors


def save_result(path: Path, rms, k, dist, size, errors, args, factory=None) -> None:
    result = {
        "image_width_px": size[0],
        "image_height_px": size[1],
        "fx_px": float(k[0, 0]),
        "fy_px": float(k[1, 1]),
        "cx_px": float(k[0, 2]),
        "cy_px": float(k[1, 2]),
        "distortion_model": "plumb_bob",
        "distortion_k1_k2_p1_p2_k3": [float(x) for x in dist.ravel()[:5]],
        "rms_reprojection_error_px": float(rms),
        "per_view_error_px": errors,
        "views": len(errors),
        "pattern_inner_corners": list(parse_pair(args.pattern, "x")),
        "square_mm": args.square_mm,
        "source": args.video or ("realsense" if args.realsense else args.device),
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if factory:
        result["factory_intrinsics"] = factory
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"\nCalibrazione salvata in {path}")
    print(f"  RMS riproiezione: {rms:.3f} px   (buono: < 0.5, accettabile: < 1.0)")
    print("  Da copiare in config/surgical_environment.yaml -> camera_overhead:")
    print(f"    intrinsics: {{fx_px: {k[0, 0]:.2f}, fy_px: {k[1, 1]:.2f}, cx_px: {k[0, 2]:.2f}, cy_px: {k[1, 2]:.2f}}}")
    print(f"    width_px: {size[0]}\n    height_px: {size[1]}")
    if factory:
        print(f"  Confronto con le intrinseche di fabbrica: fx {factory['fx_px']:.2f}, fy {factory['fy_px']:.2f}, "
              f"cx {factory['cx_px']:.2f}, cy {factory['cy_px']:.2f}")
    worst = int(np.argmax(errors))
    if max(errors) > max(2 * float(rms), 1.0):
        print(f"  Attenzione: la posa n.{worst + 1} ha errore {errors[worst]:.2f} px; annullala o rifai la cattura.")


def coverage_text(img_pts, size) -> str:
    """Quota dell'immagine (griglia 4x3) toccata dagli angoli rilevati finora."""
    if not img_pts:
        return "copertura 0%"
    cells = set()
    for pts in img_pts:
        for x, y in pts.reshape(-1, 2):
            cells.add((min(3, int(4 * x / size[0])), min(2, int(3 * y / size[1]))))
    return f"copertura {100 * len(cells) // 12}%"


def put(img, text, y, color=(255, 255, 255)):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="/dev/video4", help="camera V4L2 (default: colore della RealSense D456)")
    ap.add_argument("--video", help="calibra da un file video invece che dalla camera")
    ap.add_argument("--realsense", action="store_true", help="legge il colore della RealSense con pyrealsense2")
    ap.add_argument("--size", default="1280x720", help="risoluzione richiesta alla camera")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--pattern", default="9x6", help="angoli interni della scacchiera, colonne x righe")
    ap.add_argument("--square-mm", type=float, default=25.0, help="lato di un quadrato in mm")
    ap.add_argument("--video-step", type=int, default=15, help="con --video: un frame utile ogni N")
    ap.add_argument("--out", default="data/calibration/camera_intrinsics.json")
    ap.add_argument("--record-dir", default="data/calibration", help="dove salvare il video con il tasto V")
    ap.add_argument("--no-gui", action="store_true", help="solo con --video: nessuna finestra")
    args = ap.parse_args()
    if args.video and args.realsense:
        ap.error("--video e --realsense sono alternativi")

    pattern = parse_pair(args.pattern, "x")
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * args.square_mm

    src = RealSenseSource(args) if args.realsense else OpenCVSource(args)
    size = src.size
    print(f"Sorgente aperta: {size[0]}x{size[1]}. Scacchiera {pattern[0]}x{pattern[1]} angoli, quadrato {args.square_mm} mm.")
    gui = not (args.video and args.no_gui)
    obj_pts: list[np.ndarray] = []
    img_pts: list[np.ndarray] = []
    last = None  # (k, dist, rms)
    undistort = False
    writer = None
    last_capture = 0.0
    message, message_until = "", 0.0
    frame_index = 0

    while True:
        ok, frame = src.read()
        if not ok:
            if args.video:
                break
            if gui:
                blank = np.zeros((size[1], size[0], 3), np.uint8)
                put(blank, "nessun frame dalla camera (diagnosi: scripts/realsense_probe.py)", 30, (80, 80, 255))
                cv2.imshow("Calibrazione camera", blank)
                if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
                    break
            else:
                time.sleep(0.05)
            continue
        frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray, pattern)
        sharp = sharpness(gray)

        if args.video and corners is not None and frame_index % args.video_step == 0:
            obj_pts.append(objp)
            img_pts.append(corners)
        if writer is not None:
            writer.write(frame)
        if not gui:
            continue

        view = frame.copy()
        if undistort and last is not None:
            view = cv2.undistort(view, last[0], last[1])
        if corners is not None:
            cv2.drawChessboardCorners(view, pattern, corners, True)
        put(view, f"pose catturate: {len(img_pts)}  {coverage_text(img_pts, size)}  nitidezza {sharp:.0f}", 24)
        if corners is None:
            put(view, "scacchiera NON rilevata", 50, (80, 80, 255))
        else:
            put(view, "scacchiera rilevata: SPAZIO per catturare", 50, (80, 255, 80))
        if last is not None:
            put(view, f"ultima calibrazione RMS {last[2]:.3f} px  (D = anteprima corretta)", 76, (80, 220, 255))
        if writer is not None:
            put(view, "REC", 102, (80, 80, 255))
        if time.time() < message_until:
            put(view, message, size[1] - 14, (80, 220, 255))
        cv2.imshow("Calibrazione camera", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            if corners is None:
                message, message_until = "niente da catturare: scacchiera non rilevata", time.time() + 2
            elif time.time() - last_capture < 0.5:
                pass
            else:
                obj_pts.append(objp)
                img_pts.append(corners)
                last_capture = time.time()
                message, message_until = f"posa {len(img_pts)} catturata", time.time() + 1.5
        elif key == ord("u") and img_pts:
            obj_pts.pop()
            img_pts.pop()
            message, message_until = "ultima posa annullata", time.time() + 1.5
        elif key == ord("d"):
            undistort = not undistort
        elif key == ord("v"):
            if writer is None:
                Path(args.record_dir).mkdir(parents=True, exist_ok=True)
                name = Path(args.record_dir) / time.strftime("calib_video_%Y%m%d_%H%M%S.mp4")
                writer = cv2.VideoWriter(str(name), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
                message, message_until = f"registro in {name}", time.time() + 3
            else:
                writer.release()
                writer = None
                message, message_until = "registrazione fermata", time.time() + 2
        elif key == ord("c"):
            if len(img_pts) < 8:
                message, message_until = f"servono almeno 8 pose (ora {len(img_pts)}), meglio 20+", time.time() + 3
            else:
                rms, k, dist, errors = calibrate(obj_pts, img_pts, size)
                last = (k, dist, rms)
                save_result(Path(args.out), rms, k, dist, size, errors, args, src.factory)
                message, message_until = f"calibrato: RMS {rms:.3f} px, salvato", time.time() + 4

    if writer is not None:
        writer.release()
    src.release()
    cv2.destroyAllWindows()
    if args.video and len(img_pts) >= 8:
        rms, k, dist, errors = calibrate(obj_pts, img_pts, size)
        save_result(Path(args.out), rms, k, dist, size, errors, args, src.factory)
    elif args.video:
        print(f"Poche pose utili nel video ({len(img_pts)}): servono almeno 8.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
