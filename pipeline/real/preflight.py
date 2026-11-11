"""Preflight della pipeline reale: verifica, SENZA muovere nulla, che camera, TF, calibrazione e software siano pronti.

Da lanciare prima di ogni prova con i dispositivi fisici:

    python3 -m pipeline.real.preflight --source ros2 --ee-frame lbr_link_ee

Per ogni controllo stampa OK / WARN / FAIL con il rimedio, scrive ``real_preflight.json`` ed esce con codice 1 se
c'e' almeno un FAIL. ``ready_for_motion`` dice se una cattura adesso risulterebbe ``motion_safe`` (profondita'
misurata, intrinseche calibrate, estrinseche verificate) e con la posa dell'end-effector reale.
"""

from __future__ import annotations

import logging
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from pipeline.common import paths
from pipeline.common.cli import setup_logging, stage_parser
from pipeline.common.report import StageError, StageReport, stage_run
from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import RobotBaseFrame
from pipeline.real import ros_camera

STAGE = "real_preflight"
log = logging.getLogger(STAGE)
OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"
DEPTH_PLAUSIBILITY = 0.25        # scarto relativo tollerato tra profondita' misurata e attesa dal YAML


class Checks:
    def __init__(self):
        self.items: List[Dict[str, str]] = []

    def add(self, status: str, name: str, detail: str, fix: str = "") -> None:
        self.items.append({"status": status, "name": name, "detail": detail, "fix": fix})

    def has(self, status: str) -> bool:
        return any(c["status"] == status for c in self.items)


def check_config(checks: Checks, config: Dict[str, Any], ros: Dict[str, Any], ee_from_tf: bool) -> Dict[str, bool]:
    cam = config.get("camera_overhead", {})
    verified = bool(cam.get("extrinsics", {}).get("verified", False))
    if verified:
        checks.add(OK, "config: estrinseche verificate", "camera_overhead.extrinsics.verified: true")
    else:
        checks.add(WARN, "config: estrinseche NON verificate",
                   "camera_overhead.extrinsics.verified e' false: ogni presa resta NOT_READY_FOR_ROBOT_MOTION",
                   "dopo la calibrazione mano-occhio: pubblicala come TF e metti verified: true")
    configured_ee = bool((config.get("motion_control", {}).get("ee_reference_pose") or {}).get("position_xyz_m"))
    if ee_from_tf:
        checks.add(OK, "posa end-effector", f"letta dalla TF ({ros['world_frame']} -> {ros['ee_frame']}) a ogni cattura")
    elif configured_ee:
        checks.add(WARN, "posa end-effector", "fissa da motion_control.ee_reference_pose (non segue il robot)",
                   "per la presa reale preferire --ee-frame (TF)")
    else:
        checks.add(WARN, "posa end-effector", "nominale: la presa resta NOT_READY_FOR_ROBOT_MOTION",
                   "passare --ee-frame <frame flangia/TCP> oppure impostare motion_control.ee_reference_pose")
    base = RobotBaseFrame.from_config(config)
    identity = bool(np.allclose(base.position, 0) and np.allclose(base.rotation, np.eye(3)))
    if ros["world_frame"] == base.frame_id and not identity:
        checks.add(FAIL, "frame: world e base coincidono ma kuka_base non e' l'identita'",
                   f"world_frame='{ros['world_frame']}' e' il frame della base, ma environment.kuka_base ha una posa",
                   "con il world uguale alla base lasciare kuka_base in posizione e rotazione nulle")
    else:
        checks.add(OK, "frame: base e world coerenti",
                   f"base '{base.frame_id}' in world '{ros['world_frame']}' ({'identita' if identity else 'posa da YAML'})")
    return {"extrinsics_verified": verified, "ee_pose_available": ee_from_tf or configured_ee}


def check_software(checks: Checks, grasp_backend: str, graspgen_config: Optional[str]) -> None:
    try:
        from huggingface_hub import try_to_load_from_cache
        from pipeline.common.vision import SAM3_MODEL_ID
        cached = try_to_load_from_cache(SAM3_MODEL_ID, "config.json")
        if isinstance(cached, str):
            checks.add(OK, "SAM 3: pesi in cache", f"{SAM3_MODEL_ID} disponibile offline")
        else:
            checks.add(FAIL, "SAM 3: pesi non in cache", f"{SAM3_MODEL_ID} non scaricato (il caricamento e' offline)",
                       "scaricare i pesi con l'accesso Hugging Face prima della prova, oppure --vision/--backend diagnostico")
    except ImportError:
        checks.add(WARN, "SAM 3", "huggingface_hub non installato: impossibile controllare i pesi")
    from pipeline.plan import candidate_urls
    reachable = None
    for url in candidate_urls(None):
        host_port = url.split("//")[1].split("/")[0]
        host, port = host_port.split(":")
        try:
            with socket.create_connection((host, int(port)), timeout=1.0):
                reachable = host_port
                break
        except OSError:
            continue
    if reachable:
        checks.add(OK, "Ollama", f"raggiungibile su {reachable}")
    else:
        checks.add(WARN, "Ollama", "non raggiungibile: il planner usera' il parser deterministico",
                   "avviare Ollama sull'host oppure usare --offline (la frase deve essere semplice)")
    if grasp_backend == "graspgen":
        cfg_path = Path(graspgen_config or paths.project_root() / "config" / "graspgen.yaml")
        try:
            server = yaml.safe_load(cfg_path.read_text())["server"]
            with socket.create_connection((server["host"], int(server["port"])), timeout=2.0):
                checks.add(OK, "GraspGen: server", f"raggiungibile su {server['host']}:{server['port']}")
        except (OSError, KeyError, FileNotFoundError) as exc:
            checks.add(FAIL, "GraspGen: server", f"non raggiungibile ({exc})",
                       "avviare il server sul PC GPU (docs/graspgen.md) o aprire il tunnel SSH")
    else:
        checks.add(SKIP, "GraspGen", "backend PCA selezionato")


def check_camera(checks: Checks, config: Dict[str, Any], ros: Dict[str, Any]) -> Dict[str, Any]:
    state = {"camera_ok": False, "tf_camera": False, "ee_tf": False}
    try:
        frame = ros_camera.acquire(ros)
    except StageError as exc:
        checks.add(FAIL, "camera ROS 2", str(exc).split("] ", 1)[-1],
                   "avviare il driver della camera (profondita' allineata al colore) e controllare i nomi dei topic")
        return state
    h, w = frame.rgb.shape[:2]
    checks.add(OK, "camera: immagini", f"{w}x{h}, colore e profondita' allineati, scarto {abs(frame.color_stamp - frame.depth_stamp) * 1000:.0f} ms, "
                                       f"mediana su {frame.depth_frames_used} fotogrammi")
    checks.add(OK if frame.depth_valid_fraction >= 0.5 else WARN, "camera: profondita' valida",
               f"{frame.depth_valid_fraction:.0%} dei pixel con profondita' valida",
               "" if frame.depth_valid_fraction >= 0.5 else "controllare illuminazione, distanza minima e superfici riflettenti")
    intr = frame.intrinsics
    checks.add(OK, "camera: intrinseche (CameraInfo)", f"fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f}")
    if frame.distortion_nonzero:
        checks.add(WARN, "camera: distorsione", "CameraInfo riporta distorsione ma la pipeline usa un modello pinhole",
                   "usare image_rect_color / un flusso gia' rettificato")
    if frame.camera_pose_world is not None:
        pos = ", ".join(f"{v:.3f}" for v in frame.camera_pose_world[0])
        checks.add(OK, "TF: camera", f"{frame.world_frame} -> {frame.camera_frame}: posizione [{pos}] m")
        state["tf_camera"] = True
        expected = frame.camera_pose_world[0][2] - config["environment"]["trays"][next(iter(config["environment"]["trays"]))]["position_xyz_m"][2]
        centre = frame.depth_m[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
        centre = centre[centre > 0]
        if centre.size and abs(float(centre.mean()) - expected) > DEPTH_PLAUSIBILITY * expected:
            checks.add(WARN, "coerenza profondita' / posa camera",
                       f"profondita' media al centro {centre.mean():.2f} m, attesa circa {expected:.2f} m dalla posa TF",
                       "verificare la calibrazione mano-occhio, l'altezza dei vassoi nel YAML e che la camera guardi il vassoio")
    else:
        checks.add(FAIL, "TF: camera", f"manca {frame.world_frame} -> {frame.camera_frame}",
                   "pubblicare la calibrazione mano-occhio come TF (static_transform_publisher) o correggere i nomi dei frame")
    if ros.get("ee_frame"):
        if frame.ee_pose_world is not None:
            checks.add(OK, "TF: end-effector", f"{frame.world_frame} -> {ros['ee_frame']} disponibile")
            state["ee_tf"] = True
        else:
            checks.add(FAIL, "TF: end-effector", f"manca {frame.world_frame} -> {ros['ee_frame']}",
                       "avviare il driver del robot (lbr_fri_ros2_stack) o correggere --ee-frame")
    state["camera_ok"] = True
    state["measured_depth"] = frame.depth_valid_fraction >= 0.5
    return state


def run(*, source: str = "ros2", ros: Optional[dict] = None, config_path: Optional[str] = None,
        grasp_backend: str = "pca", graspgen_config: Optional[str] = None, out_dir: Optional[str] = None,
        report_path: Optional[str] = None) -> StageReport:
    out = paths.output_dir(out_dir)
    with stage_run(STAGE, paths.output_report(report_path, "real_preflight.json", out),
                   inputs={"config": config_path}, logger=log) as report:
        config = load_environment_config(config_path)
        options = ros_camera.resolve_options(config, ros)
        checks = Checks()
        state: Dict[str, Any] = {}
        if source == "ros2":
            state.update(check_camera(checks, config, options))
        else:
            checks.add(SKIP, "camera ROS 2", "sorgente 'file': nessuna camera da controllare")
        config_state = check_config(checks, config, options, ee_from_tf=bool(state.get("ee_tf")))
        check_software(checks, grasp_backend, graspgen_config)
        ready = (source == "ros2" and state.get("camera_ok") and state.get("tf_camera") and state.get("measured_depth")
                 and config_state["extrinsics_verified"] and config_state["ee_pose_available"]
                 and not checks.has(FAIL))
        report.data = {"source": source, "checks": checks.items, "ready_for_motion": bool(ready),
                       "ros2_options": options}
        for item in checks.items:
            if item["status"] in (WARN, FAIL):
                report.warn(log, f"{item['name']}: {item['detail']}")
        report.data["failed"] = checks.has(FAIL)
    return report


def print_summary(report: StageReport) -> None:
    mark = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[skip]"}
    print("\nPreflight pipeline reale")
    print("=" * 60)
    for item in report.data["checks"]:
        print(f"{mark[item['status']]} {item['name']}: {item['detail']}")
        if item["fix"] and item["status"] in (WARN, FAIL):
            print(f"        -> {item['fix']}")
    print("=" * 60)
    if report.data["failed"]:
        print("ESITO: NON pronto (ci sono FAIL).")
    elif report.data["ready_for_motion"]:
        print("ESITO: pronto. Una cattura adesso risulterebbe motion_safe.")
    else:
        print("ESITO: acquisizione possibile, ma la presa NON sarebbe abilitata al movimento (vedi WARN).")


def main(argv: Optional[List[str]] = None) -> int:
    from pipeline.real import capture
    parser = stage_parser(__doc__.splitlines()[0])
    capture.add_source_arguments(parser)
    parser.set_defaults(source="ros2")
    parser.add_argument("--grasp-backend", choices=("pca", "graspgen"), default="pca")
    parser.add_argument("--graspgen-config", default=None)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    try:
        report = run(source=args.source, ros=capture.source_kwargs(args)["ros"], config_path=args.config,
                     grasp_backend=args.grasp_backend, graspgen_config=args.graspgen_config, out_dir=args.out_dir,
                     report_path=args.report)
    except StageError as exc:
        logging.getLogger("cli").error("%s", exc)
        return 1
    print_summary(report)
    return 1 if report.data["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
