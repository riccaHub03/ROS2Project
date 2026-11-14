"""Independent launch modes for diagnostic GraspGen and the complete AI system."""
import argparse
import json
import os
from pathlib import Path

from pipeline.common.cli import setup_logging
from pipeline.system.registry import load_profile
from pipeline.system.motion import motion_speed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("graspgen_only", "full_system"), required=True)
    parser.add_argument("--instruction", help="Natural-language pick-and-place task")
    parser.add_argument("--tool", help="Configured scene tool; does not override a full-system LLM response")
    parser.add_argument("--config", help="Environment YAML with an explicit CAD mesh")
    parser.add_argument("--ollama-url", help="Chat Completions URL of the local Ollama service")
    parser.add_argument("--llm-model")
    parser.add_argument("--sam3-model", help="Cached HuggingFace ID or local model directory (CPU)")
    parser.add_argument("--check", action="store_true", help="Check services/assets without inference or motion")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--gui", action="store_true", help="Show PyBullet (default)")
    display.add_argument("--headless", action="store_true")
    parser.add_argument("--exit-after-run", action="store_true")
    parser.add_argument("--out-dir", help="New output directory; must not already exist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--motion-speed", type=motion_speed, default=1.,
                        help="Simulation speed reference factor, 0.25–2 (default 1); dynamics still checked")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib"))
    setup_logging(args.verbose)
    if args.config and not args.tool:
        parser.error("For a custom scene, supply both --config and --tool to identify its CAD asset")
    if args.tool and not args.config and args.tool.casefold() != "bone forceps":
        parser.error("A different tool requires --config with its CAD scene; default scene contains Bone Forceps")
    try:
        settings = load_profile(args.profile, environment=args.config, scene_tool=args.tool,
                                ollama_url=args.ollama_url, llm_model=args.llm_model, sam3_model=args.sam3_model)
        settings["motion_speed"] = args.motion_speed
        if args.check:
            from pipeline.system.preflight import check
            results = check(settings)
            print(json.dumps(results, indent=2))
            return 0 if all(r["ok"] for r in results) else 1
        args.instruction = args.instruction or f"Pick up the {settings['scene_tool']} from tray 2 and place it in tray 1."
        from pipeline.system.orchestrator import run
        return run(settings, args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"System configuration error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
