"""Stage ``plan``: natural-language instruction -> structured task plan (Qwen 2.5 via Ollama).

Input : instruction text (CLI/arguments) + environment config (tray aliases)
Output: ``plan.json`` -> data: action, target_instrument, target_keyword, source_tray,
        destination_tray, planner_backend, instruction

Run alone:  python -m pipeline.plan --instruction "Pass me the scalpel from tray 2"
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional

from pipeline.common import paths
from pipeline.common.cli import run_cli, stage_parser
from pipeline.common.report import StageReport, stage_run
from pipeline.common.scene import Config
from pipeline.config_loader import load_environment_config

STAGE = "plan"
log = logging.getLogger(STAGE)

DEFAULT_INSTRUCTION = "Passami le forbici vascolari posizionate sul vassoio 2 e mettile nel vassoio 1 per la sutura."
OLLAMA_PORT = 11434
OLLAMA_PATH = "/v1/chat/completions"
DEFAULT_MODEL = "qwen2.5:3b"
REQUEST_TIMEOUT_S = 12

SYSTEM_PROMPT = (
    "You are an expert AI surgical robotics task planner.\n"
    "Parse the clinical user instruction and output a STRICT JSON object with these EXACT keys:\n"
    "  - 'action': 'pick_and_place' | 'pick' | 'inspect'\n"
    "  - 'target_instrument': precise English name of the surgical tool (e.g. 'vascular scissors', 'scalpel', 'hemostatic forceps')\n"
    "  - 'target_keyword': single lowercase keyword for file/model matching (e.g. 'scissors', 'scalpel', 'forceps', 'clamp')\n"
    "  - 'source_tray': 'tray_2' (sterile supply tray) or 'tray_1' or 'table'\n"
    "  - 'destination_tray': 'tray_1' (collection tray) or 'tray_2' or 'table'\n\n"
    "Example output:\n"
    "{\n"
    "  \"action\": \"pick_and_place\",\n"
    "  \"target_instrument\": \"vascular scissors\",\n"
    "  \"target_keyword\": \"scissors\",\n"
    "  \"source_tray\": \"tray_2\",\n"
    "  \"destination_tray\": \"tray_1\"\n"
    "}\n"
    "Return ONLY the valid JSON object, without any Markdown code blocks or explanations."
)

# (substring found in the instrument name, canonical keyword)
KEYWORD_ROOTS = [
    ("scalpel", "scalpel"), ("scissor", "scissors"), ("forcep", "forceps"), ("clamp", "clamp"),
    ("needle", "needle"), ("retractor", "retractor"), ("curette", "curette"), ("chisel", "chisel"),
    ("mallet", "mallet"), ("drill", "drill"), ("tweezer", "tweezer"), ("spatula", "spatula"),
    ("raspatory", "raspatory"), ("probe", "probe"), ("mirror", "mirror"), ("gag", "gag"), ("burr", "burr"),
    ("osteotome", "osteotome"), ("speculum", "speculum"), ("marker", "marker"), ("tamper", "tamper"),
    ("bisturi", "scalpel"), ("pinza", "forceps"),
]
STOP_WORDS = {"the", "a", "an", "surgical", "instrument", "tool", "lo", "strumento"}


# --------------------------------------------------------------------------- normalization
def extract_keyword(instrument_name: str) -> str:
    """Clinical keyword (e.g. 'scissors') from a free-form instrument description."""
    lowered = instrument_name.lower()
    for root, keyword in KEYWORD_ROOTS:
        if root in lowered:
            return keyword
    words = [w for w in re.findall(r"\w+", lowered) if w not in STOP_WORDS]
    return words[0] if words else lowered


def normalize_tray(value: Any, default: str, config: Optional[Config] = None) -> str:
    if not value:
        return default
    text = str(value).lower().strip()
    for key, info in (config or {}).get("environment", {}).get("trays", {}).items():
        if key.lower() == text:
            return key
        if any(alias.lower() in text or text in alias.lower() for alias in info.get("alias", [])):
            return key
    if any(k in text for k in ("2", "sinistro", "sterile", "left")):
        return "tray_2"
    if any(k in text for k in ("1", "destro", "decontaminazione", "raccolta", "right")):
        return "tray_1"
    return default


def normalize_action(value: Any) -> str:
    text = str(value or "").lower()
    if any(k in text for k in ("pick", "place", "move", "prendi", "sposta", "passa")):
        return "pick_and_place"
    return "inspect" if "inspect" in text else "pick_and_place"


# --------------------------------------------------------------------------- planner backends
def candidate_urls(explicit: Optional[str]) -> List[str]:
    if explicit:
        return [explicit]
    urls = []
    host = os.environ.get("OLLAMA_HOST")
    if host:
        urls.append(f"http://{host}:{OLLAMA_PORT}{OLLAMA_PATH}")
    urls += [f"http://{h}:{OLLAMA_PORT}{OLLAMA_PATH}" for h in ("172.19.128.1", "localhost", "127.0.0.1")]
    return list(dict.fromkeys(urls))


def parse_llm_json(raw: str) -> Dict[str, Any]:
    """Strip <think> blocks and Markdown fences, then parse the JSON object."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def query_ollama(instruction: str, urls: List[str], model: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    payload = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 120, "messages": [
        {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": instruction}]}).encode()
    for url in urls:
        try:
            request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as resp:
                content = json.loads(resp.read().decode())["choices"][0]["message"]["content"]
            return parse_llm_json(content.strip()), url
        except Exception as exc:  # network or malformed answer: try the next candidate
            log.debug("Ollama at %s unusable: %s", url, exc)
    return None, None


def deterministic_plan(instruction: str) -> Dict[str, str]:
    """Keyword-based fallback used offline or when no Ollama server answers."""
    text = instruction.lower()
    if "bisturi" in text or "scalpel" in text:
        tool, keyword = "scalpel", "scalpel"
    elif "pinza" in text or "forceps" in text:
        tool, keyword = "hemostatic forceps", "forceps"
    else:
        tool, keyword = "vascular scissors", "scissors"
    source = "tray_2" if "2" in text else "tray_1"
    return {"action": "pick_and_place", "target_instrument": tool, "target_keyword": keyword,
            "source_tray": source, "destination_tray": "tray_1" if source == "tray_2" else "tray_2"}


# --------------------------------------------------------------------------- stage
def run(instruction: str = DEFAULT_INSTRUCTION, *, tool_hint: Optional[str] = None,
        ollama_url: Optional[str] = None, model: str = DEFAULT_MODEL, offline: bool = False,
        config_path: Optional[str] = None, out_dir: Optional[str] = None,
        report_path: Optional[str] = None) -> StageReport:
    out = paths.output_report(report_path, paths.PLAN, out_dir)
    with stage_run(STAGE, out, inputs={"config": config_path}, logger=log) as report:
        log.info("instruction: %r%s", instruction, f" (tool hint: {tool_hint})" if tool_hint else "")
        parsed, url = (None, None) if offline else query_ollama(instruction, candidate_urls(ollama_url), model)
        if parsed is None:
            log.warning("deterministic parser in use (%s)", "offline requested" if offline else "no Ollama server reachable")
            parsed = deterministic_plan(instruction)
        else:
            log.info("LLM answer from %s", url)

        instrument = str(tool_hint or parsed.get("target_instrument") or "vascular scissors").strip()
        keyword = str(parsed.get("target_keyword", "")).strip().lower()
        if not keyword or (keyword == "scissors" and "scissor" not in instrument.lower()):
            keyword = extract_keyword(instrument)

        config = load_environment_config(config_path)
        report.data = {
            "instruction": instruction,
            "action": normalize_action(parsed.get("action")),
            "target_instrument": instrument,
            "target_keyword": keyword,
            "source_tray": normalize_tray(parsed.get("source_tray"), "tray_2", config),
            "destination_tray": normalize_tray(parsed.get("destination_tray"), "tray_1", config),
            "planner_backend": "ollama" if url else "deterministic_parser",
        }
        log.info("plan: %s", report.data)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = stage_parser(__doc__.splitlines()[0])
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="Operator command in natural language")
    parser.add_argument("--tool", default=None, help="Force the target instrument name (overrides the LLM)")
    parser.add_argument("--offline", action="store_true", help="Skip Ollama, use the deterministic parser")
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return run_cli(parser, lambda a: run(a.instruction, tool_hint=a.tool, ollama_url=a.ollama_url, model=a.model,
                                         offline=a.offline, config_path=a.config, out_dir=a.out_dir,
                                         report_path=a.report), argv)


if __name__ == "__main__":
    raise SystemExit(main())
