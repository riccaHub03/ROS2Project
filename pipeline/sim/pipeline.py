"""Simulated end-to-end pipeline: plan -> capture -> segment -> reference_pose -> deproject -> grasp.

Run:  python run_pipeline_sim.py --tool Scalpel          (or: python -m pipeline.sim.pipeline ...)
With --gui/--headless the KUKA scene is driven by ``pipeline.sim.graphical``.

By default (``--mode all``) the perception covers the WHOLE All_Instruments collection: RGB, depth picture and
a mask of every instrument. The KUKA window always shows the single STL of the target being transported; in
that case the full-collection views are produced offscreen (``sim_overview.json``).
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

from pipeline import grasp, plan
from pipeline.common import paths
from pipeline.common.cli import setup_logging
from pipeline.common.report import StageError
from pipeline.orchestrate import Step, execute
from pipeline.sim import capture, deproject, reference_pose, segment

STAGES = ["plan", "capture", "segment", "reference_pose", "deproject", "grasp"]
log = logging.getLogger("pipeline.sim")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instruction", default=None, help="Clinical command for the LLM task planner")
    parser.add_argument("--tool", default=None, help="Specific instrument, e.g. 'Scalpel', 'Bone Forceps', 'Scissors'")
    parser.add_argument("--list-tools", action="store_true", help="Print the instrument catalog and exit")
    parser.add_argument("--mode", choices=["all", "single"], default=None,
                        help="'all' (default): RGB-D and mask of EVERY instrument on the full All_Instruments scene; "
                             "'single': only the target STL. With --gui/--headless the window always shows the STL and "
                             "'all' adds the offscreen overview. Without the robot, --vision ground-truth defaults to "
                             "'single' (the simulator mask needs one body per instrument)")
    parser.add_argument("--offline", action="store_true", help="Skip Ollama: deterministic planner")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the instrument placement in the tray")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--from-stage", choices=STAGES, default=None, help="Resume from this stage (earlier reports must exist)")
    parser.add_argument("--stop-after", choices=STAGES, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--gui", action="store_true", help="Native 3D window with the KUKA and transfer preview")
    display.add_argument("--headless", action="store_true", help="Same run with the robot, no window")
    parser.add_argument("--vision", choices=["sam3", "ground-truth"], default="sam3",
                        help="SAM 3, or the diagnostic ground-truth mask rendered by the simulator (skips SAM 3)")
    parser.add_argument("--no-motion", action="store_true", help="Show the grasp without moving the robot")
    parser.add_argument("--exit-after-run", action="store_true", help="Close the window when the run ends")
    grasp.add_backend_arguments(parser)
    return parser


def print_catalog() -> None:
    from pipeline.mesh_loader import load_instruments_catalog
    catalog = load_instruments_catalog()
    if not catalog:
        print("Catalog not found. Generate it with scripts/build_instruments_catalog.py")
        return
    print(f"Categories: {catalog['total_categories']} | STL models: {catalog['total_stl_files']}\n")
    print(f"{'#':<3} | {'CATEGORY':<26} | {'MODELS':<8} | EXAMPLE")
    for idx, (name, info) in enumerate(catalog["categories"].items(), 1):
        sample = info["files"][0]["model_name"] if info["files"] else "N/A"
        print(f"{idx:<3} | {name:<26} | {info['models_count']:<8} | {sample[:30]}")
    print('\nTry: python run_pipeline_sim.py --tool "Bone Forceps"')


def resolve_mode(args: argparse.Namespace) -> str:
    """Perception scene: explicit --mode, else 'all'; the ground-truth backend without robot falls back to 'single'."""
    if args.mode:
        return args.mode
    if args.gui or args.headless:
        return "all"
    return "single" if args.vision == "ground-truth" else "all"


def build_steps(args: argparse.Namespace, instruction: str, mode: str) -> List[Step]:
    common = dict(out_dir=args.out_dir)
    cfg = dict(config_path=args.config, **common)
    out = paths.output_dir(args.out_dir)
    return [
        Step("plan", lambda: plan.run(instruction, tool_hint=args.tool, offline=args.offline, **cfg)),
        Step("capture", lambda: capture.run(mode=mode, **cfg)),
        Step("segment", lambda: segment.run(backend=args.vision, all_instruments=(mode == "all"), **common)),
        Step("reference_pose", lambda: reference_pose.run(tool=args.tool, seed=args.seed, **cfg)),
        Step("deproject", lambda: deproject.run(reference_pose_path=str(out / paths.SIM_REFERENCE_POSE), **cfg)),
        Step("grasp", lambda: grasp.run(deprojection_path=str(out / paths.SIM_DEPROJECTION),
                                       **grasp.backend_kwargs(args), **cfg)),
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if args.list_tools:
        print_catalog()
        return 0

    instruction = args.instruction or (
        f"Passami lo strumento {args.tool} posizionato sul vassoio 2 e mettilo nel vassoio 1 per l'intervento chirurgico."
        if args.tool else plan.DEFAULT_INSTRUCTION)
    robot_run = args.gui or args.headless
    mode = resolve_mode(args)
    if robot_run:
        args.mode = mode        # the robot scene is always the STL; 'all' adds the offscreen overview
        from pipeline.sim.graphical import run_graphical_pipeline
        return run_graphical_pipeline(args, instruction)
    if args.no_motion or args.exit_after_run:
        parser.error("--no-motion and --exit-after-run require --gui or --headless")
    try:
        execute(build_steps(args, instruction, mode), start_at=args.from_stage, stop_after=args.stop_after)
    except StageError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
