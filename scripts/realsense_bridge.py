#!/usr/bin/env python3
"""Ponte dal vivo per la RealSense: camera su Windows -> topic ROS 2 nel container.

Su un PC Windows con WSL la camera USB funziona solo su Windows, mentre ROS 2 e la pipeline stanno nel container.
Il ponte porta i fotogrammi dal vivo dall'uno all'altro sulla rete interna (localhost, ~600 MB/s misurati):

  Windows    send   camera -> pyrealsense2 -> colore + profondita' allineata al colore --TCP--+
  container  ros    li riceve e li pubblica come un driver camera  <--------------------------+
                    /camera/color/image_raw (bgr8)   /camera/aligned_depth_to_color/image_raw (16UC1, mm)
                    /camera/color/camera_info        /camera/aligned_depth_to_color/camera_info

Cosi' `pipeline.real --source ros2` (preflight e cattura) acquisisce dal vivo senza modifiche:

  python3 scripts/realsense_bridge.py ros                              # nel container, con ROS 2 caricato
  bash scripts/rs_windows.sh scripts/realsense_bridge.py send          # in WSL: Python di Windows, vede la camera
  bash scripts/rs_windows.sh scripts/realsense_bridge.py send --fake   # scena sintetica, senza camera

L'ordine di avvio non conta: il mittente riprova finche' il ricevitore non c'e' e si ricollega se cade. I messaggi
hanno l'istante di arrivo nel container (colore e profondita' arrivano insieme: scarto zero). Il ponte NON pubblica la
TF world -> camera: viene dalla calibrazione mano-occhio (vedi README). Guida: docs/realsense.md
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time

PORT = 50551
PROTOCOL = "rs-bridge/1"
MAGIC = b"RSF1"
FRAME = struct.Struct("<4sQdII")  # magic, numero, istante di acquisizione (s, orologio del mittente), byte colore, byte profondita'


# ------------------------------------------------------------------------------------------------ protocollo
def send_json(sock: socket.socket, payload: dict) -> None:
    data = json.dumps(payload).encode()
    sock.sendall(struct.pack("<I", len(data)) + data)


def recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("connessione chiusa dal mittente")
        got += k
    return buf


def recv_json(sock: socket.socket) -> dict:
    (n,) = struct.unpack("<I", recv_exact(sock, 4))
    if not 0 < n < 1_000_000:
        raise ConnectionError("intestazione non valida (non e' un mittente del ponte?)")
    return json.loads(bytes(recv_exact(sock, n)))


def parse_profile(text: str) -> tuple[int, int, int]:
    size, fps = text.lower().split("@")
    w, h = size.split("x")
    return int(w), int(h), int(fps)


# ------------------------------------------------------------------------------------------------ sorgenti (Windows)
def open_camera(args):
    """Apre la RealSense: colore + profondita' allineata al colore. Ritorna (intestazione, grab, close)."""
    import numpy as np
    import pyrealsense2 as rs

    devs = rs.context().query_devices()
    if not len(devs):
        sys.exit("Nessuna RealSense trovata: collegala, chiudi il RealSense Viewer (diagnosi: scripts/realsense_probe.py)")
    dev = devs[0]
    info = {k: dev.get_info(getattr(rs.camera_info, k)) if dev.supports(getattr(rs.camera_info, k)) else "?"
            for k in ("name", "serial_number", "usb_type_descriptor")}

    def profiles(stream, fmt):
        found = set()
        for sensor in dev.query_sensors():
            for p in sensor.get_stream_profiles():
                if p.stream_type() == stream and p.format() == fmt and p.is_video_stream_profile():
                    v = p.as_video_stream_profile()
                    found.add((v.width(), v.height(), p.fps()))
        return sorted(found)

    def pick(avail, w, h, fps):
        return min(avail, key=lambda p: (abs(p[0] - w) + abs(p[1] - h), abs(p[2] - fps))) if avail else None

    w, h, fps = parse_profile(args.color)
    c = pick(profiles(rs.stream.color, rs.format.bgr8), w, h, fps)
    depth_avail = profiles(rs.stream.depth, rs.format.z16)
    d = pick([p for p in depth_avail if c and p[2] == c[2]] or depth_avail, w, h, c[2] if c else fps)
    if c is None or d is None:
        sys.exit(f"La camera non offre colore e profondita' insieme con il collegamento USB {info['usb_type_descriptor']}: "
                 "serve una porta USB 3 (scripts/realsense_probe.py)")
    cfg = rs.config()
    cfg.enable_device(info["serial_number"])
    cfg.enable_stream(rs.stream.color, c[0], c[1], rs.format.bgr8, c[2])
    cfg.enable_stream(rs.stream.depth, d[0], d[1], rs.format.z16, d[2])
    pipe = rs.pipeline()
    try:
        profile = pipe.start(cfg)
    except RuntimeError as exc:
        sys.exit(f"Impossibile avviare la camera ({exc}): e' aperta in un altro programma?")
    align = rs.align(rs.stream.color)
    i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    header = {
        "proto": PROTOCOL, "width": i.width, "height": i.height, "fps": c[2],
        "fx": i.fx, "fy": i.fy, "cx": i.ppx, "cy": i.ppy, "coeffs": list(i.coeffs)[:5],
        "depth_scale": profile.get_device().first_depth_sensor().get_depth_scale(),
        "name": info["name"], "serial": info["serial_number"], "usb": info["usb_type_descriptor"],
        "depth_profile": list(d),
    }

    def grab():
        frames = align.process(pipe.wait_for_frames(5000))
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            return None
        return (color.get_frame_number(), np.ascontiguousarray(np.asanyarray(color.get_data())),
                np.ascontiguousarray(np.asanyarray(depth.get_data())))

    return header, grab, pipe.stop


def open_fake(args):
    """Scena sintetica senza camera: vassoio piano a 1,06 m con uno strumento piu' alto di 12 mm (come
    scripts/fake_ros2_camera.py) e un quadratino che si muove, per vedere che le immagini sono dal vivo."""
    import numpy as np

    w, h, fps = parse_profile(args.color)
    fx = 615.0 * w / 640.0
    base_bgr = np.full((h, w, 3), (186, 182, 180), np.uint8)
    depth_mm = np.full((h, w), 1060, np.uint16)
    y0, y1, x0, x1 = h // 2 - h // 24, h // 2 + h // 24, w // 2 - w // 6, w // 2 + w // 6
    base_bgr[y0:y1, x0:x1] = (84, 76, 70)
    depth_mm[y0:y1, x0:x1] = 1048
    header = {"proto": PROTOCOL, "width": w, "height": h, "fps": fps, "fx": fx, "fy": fx, "cx": w / 2, "cy": h / 2,
              "coeffs": [0.0] * 5, "depth_scale": 0.001, "name": "Scena sintetica", "serial": "fake", "usb": "-",
              "depth_profile": [w, h, fps]}
    state = {"n": 0, "next": time.time()}

    def grab():
        # un fotogramma ogni 1/fps; se si e' rimasti indietro (ricevitore assente) si riparte da adesso,
        # senza raffiche per recuperare i fotogrammi persi (come fa una camera vera)
        now = time.time()
        state["next"] = max(state["next"] + 1.0 / fps, now)
        time.sleep(state["next"] - now)
        n = state["n"] = state["n"] + 1
        bgr = base_bgr.copy()
        x = int((n * 4) % (w - 20))
        bgr[10:30, x : x + 20] = (0, 0, 255)
        return n, bgr, depth_mm

    return header, grab, lambda: None


# ------------------------------------------------------------------------------------------------ mittente (Windows)
def run_send(args) -> int:
    header, grab, close = (open_fake if args.fake else open_camera)(args)
    print(f"{header['name']} {header['serial']} (USB {header['usb']}): colore {header['width']}x{header['height']}@"
          f"{header['fps']}, profondita' {'x'.join(map(str, header['depth_profile'][:2]))} allineata al colore.")
    print(f"Invio a {args.host}:{args.port}. Ctrl+C per fermare.")
    waiting = False
    try:
        while True:
            try:
                sock = socket.create_connection((args.host, args.port), timeout=2)
            except OSError:
                if not waiting:
                    print("In attesa del ricevitore nel container: python3 scripts/realsense_bridge.py ros")
                    waiting = True
                time.sleep(1)
                continue
            waiting = False
            sock.settimeout(10)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("Collegato: trasmissione dal vivo.")
            sent, t_stats = 0, time.time()
            try:
                send_json(sock, header)
                while True:
                    try:
                        item = grab()
                    except RuntimeError as exc:  # pyrealsense2: nessun fotogramma entro il tempo limite
                        print(f"La camera non manda fotogrammi ({exc}): controlla il collegamento con "
                              "scripts/realsense_probe.py")
                        return 2
                    if item is None:
                        continue
                    n, color, depth = item
                    sock.sendall(FRAME.pack(MAGIC, n, time.time(), color.nbytes, depth.nbytes))
                    sock.sendall(color)
                    sock.sendall(depth)
                    sent += 1
                    if time.time() - t_stats >= args.stats_s:
                        print(f"  inviati {sent / (time.time() - t_stats):.1f} fotogrammi/s")
                        sent, t_stats = 0, time.time()
            except OSError as exc:
                print(f"Ricevitore scollegato ({exc}): riprovo.")
            finally:
                sock.close()
    except KeyboardInterrupt:
        print("Fermato.")
    finally:
        close()
    return 0


# ------------------------------------------------------------------------------------------------ ricevitore (container)
def run_ros(args) -> int:
    import array
    import statistics

    import numpy as np
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    rclpy.init()
    node = rclpy.create_node("realsense_bridge")
    pubs = {name: node.create_publisher(kind, topic, qos_profile_sensor_data) for name, kind, topic in (
        ("color", Image, args.color_topic), ("depth", Image, args.depth_topic),
        ("info", CameraInfo, args.camera_info_topic), ("depth_info", CameraInfo, args.depth_info_topic))}
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(1)
    srv.settimeout(1.0)
    print(f"Ricevitore in ascolto su {args.bind}:{args.port}; pubblica {args.color_topic}, {args.depth_topic}, "
          f"{args.camera_info_topic} (frame {args.frame_id}).")
    print("Avvia il mittente in WSL: bash scripts/rs_windows.sh scripts/realsense_bridge.py send")
    try:
        while rclpy.ok():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(10)
            try:
                hdr = recv_json(conn)
                if hdr.get("proto") != PROTOCOL:
                    raise ConnectionError(f"protocollo {hdr.get('proto')!r}, atteso {PROTOCOL!r}")
                w, h = int(hdr["width"]), int(hdr["height"])
                info = CameraInfo(width=w, height=h, distortion_model="plumb_bob", d=[float(v) for v in hdr["coeffs"]],
                                  k=[hdr["fx"], 0.0, hdr["cx"], 0.0, hdr["fy"], hdr["cy"], 0.0, 0.0, 1.0],
                                  r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                                  p=[hdr["fx"], 0.0, hdr["cx"], 0.0, 0.0, hdr["fy"], hdr["cy"], 0.0, 0.0, 0.0, 1.0, 0.0])
                scale = float(hdr["depth_scale"])
                in_mm = abs(scale - 0.001) < 1e-9
                print(f"Mittente collegato: {hdr['name']} {hdr['serial']} (USB {hdr['usb']}), {w}x{h}@{hdr['fps']}, "
                      f"profondita' {'16UC1 in mm' if in_mm else '32FC1 in m'}.")
                count, delays, t_stats = 0, [], time.time()
                while rclpy.ok():
                    magic, _n, t_capture, n_color, n_depth = FRAME.unpack(recv_exact(conn, FRAME.size))
                    if magic != MAGIC or n_color != w * h * 3 or n_depth != w * h * 2:
                        raise ConnectionError("fotogramma non valido")
                    color, depth = recv_exact(conn, n_color), recv_exact(conn, n_depth)
                    delays.append(time.time() - t_capture)
                    color_msg = Image(height=h, width=w, encoding="bgr8", step=w * 3)
                    color_msg.data = array.array("B", color)  # array.array: assegnazione veloce (niente controllo per byte)
                    if in_mm:
                        depth_msg = Image(height=h, width=w, encoding="16UC1", step=w * 2)
                        depth_msg.data = array.array("B", depth)
                    else:
                        meters = np.frombuffer(depth, "<u2").astype(np.float32) * scale
                        depth_msg = Image(height=h, width=w, encoding="32FC1", step=w * 4)
                        depth_msg.data = array.array("B", meters.tobytes())
                    stamp = node.get_clock().now().to_msg()
                    for msg in (color_msg, depth_msg, info):
                        msg.header.stamp, msg.header.frame_id = stamp, args.frame_id
                    pubs["color"].publish(color_msg)
                    pubs["depth"].publish(depth_msg)
                    pubs["info"].publish(info)
                    pubs["depth_info"].publish(info)
                    count += 1
                    if time.time() - t_stats >= args.stats_s:
                        print(f"  pubblicati {count / (time.time() - t_stats):.1f} fotogrammi/s, ritardo mediano "
                              f"{statistics.median(delays) * 1000:.0f} ms (orologi Windows/WSL)")
                        count, delays, t_stats = 0, [], time.time()
            except (OSError, ValueError, KeyError) as exc:
                print(f"Mittente scollegato ({exc}). In attesa di una nuova connessione...")
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("Fermato.")
    finally:
        srv.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("send", help="su Windows: legge la camera e invia i fotogrammi")
    s.add_argument("--host", default="127.0.0.1", help="dove ascolta il ricevitore (da Windows: localhost)")
    s.add_argument("--port", type=int, default=PORT)
    s.add_argument("--color", default="640x480@15", help="profilo colore LxA@fps; la profondita' viene allineata a questo")
    s.add_argument("--fake", action="store_true", help="scena sintetica senza camera, per provare la catena")
    s.add_argument("--stats-s", type=float, default=10.0, help="ogni quanti secondi stampare la frequenza")
    r = sub.add_parser("ros", help="nel container: riceve i fotogrammi e li pubblica come topic ROS 2")
    r.add_argument("--bind", default="0.0.0.0", help="indirizzo di ascolto (0.0.0.0: raggiungibile da Windows via localhost)")
    r.add_argument("--port", type=int, default=PORT)
    r.add_argument("--frame-id", default="camera_color_optical_frame", help="frame OTTICO della camera")
    r.add_argument("--color-topic", default="/camera/color/image_raw")
    r.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    r.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    r.add_argument("--depth-info-topic", default="/camera/aligned_depth_to_color/camera_info")
    r.add_argument("--stats-s", type=float, default=10.0)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # messaggi subito visibili anche se l'uscita va su un file (avvio in background)
    return run_send(args) if args.mode == "send" else run_ros(args)


if __name__ == "__main__":
    sys.exit(main())
