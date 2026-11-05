#!/usr/bin/env python3
"""
Step 1: LLM Task Planner (Qwen 2.5 3B via Ollama).
Parte 1 della Catena di Montaggio Chirurgica.
Input: Istruzione in linguaggio naturale (es. dal chirurgo).
Output: data/outputs/stage1_task_plan.json
"""

import os
import sys
import json
import re
import argparse
import datetime
import urllib.request

from pipeline.config_loader import load_environment_config

DEFAULT_INSTRUCTION = "Passami le forbici vascolari posizionate sul vassoio 2 e mettile nel vassoio 1 per la sutura."

def extract_keyword_from_instrument(instrument_name):
    """Estrae la parola chiave clinica dalla descrizione dello strumento."""
    inst_lower = instrument_name.lower()
    # Mappatura radici semantiche note
    kw_roots = [
        "scalpel", "scissor", "forcep", "clamp", "needle", "retractor", "curette",
        "chisel", "mallet", "drill", "tweezer", "spatula", "raspatory", "probe",
        "mirror", "gag", "burr", "osteotome", "speculum", "marker", "tamper", "bisturi", "pinza"
    ]
    for kw in kw_roots:
        if kw in inst_lower:
            if kw == "scissor":
                return "scissors"
            if kw == "forcep":
                return "forceps"
            if kw == "bisturi":
                return "scalpel"
            if kw == "pinza":
                return "forceps"
            return kw
    words = [w for w in re.findall(r'\w+', inst_lower) if w not in ('the', 'a', 'an', 'surgical', 'instrument', 'tool', 'lo', 'lo', 'strumento')]
    return words[0] if words else inst_lower


def normalize_tray(val, default="tray_2", config=None):
    if not val:
        return default
    v = str(val).lower().strip()
    if config:
        trays = config.get("environment", {}).get("trays", {})
        for t_k, t_info in trays.items():
            if t_k.lower() == v:
                return t_k
            aliases = t_info.get("alias", [])
            for a in aliases:
                if a.lower() in v or v in a.lower():
                    return t_k
    if "2" in v or "sinistro" in v or "sterile" in v or "left" in v:
        return "tray_2"
    if "1" in v or "destro" in v or "decontaminazione" in v or "raccolta" in v or "right" in v:
        return "tray_1"
    return default


def normalize_action(val):
    if not val:
        return "pick_and_place"
    v = str(val).lower()
    if any(k in v for k in ["pick", "place", "move", "prendi", "sposta", "passa"]):
        return "pick_and_place"
    if "inspect" in v:
        return "inspect"
    return "pick_and_place"


def run_llm_planner(instruction_text=DEFAULT_INSTRUCTION, target_tool_hint=None, ollama_url=None, model="qwen2.5:3b", output_path=None, offline=False, config_path=None):
    print("\n" + "="*70)
    print("[CATENA DI MONTAGGIO - STAGE 1] LLM TASK PLANNER")
    print(f"   Istruzione ricevuta: \"{instruction_text}\"")
    if target_tool_hint:
        print(f"   Tool Hint Richiesto: {target_tool_hint}")
    print("="*70)

    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, "data", "outputs", "stage1_task_plan.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Determinazione URL Ollama
    candidate_urls = []
    if ollama_url:
        candidate_urls.append(ollama_url)
    env_host = os.environ.get("OLLAMA_HOST")
    if env_host:
        candidate_urls.append(f"http://{env_host}:11434/v1/chat/completions")
    candidate_urls.extend([
        "http://172.19.128.1:11434/v1/chat/completions",
        "http://localhost:11434/v1/chat/completions",
        "http://127.0.0.1:11434/v1/chat/completions"
    ])

    system_prompt = (
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

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction_text}
        ],
        "temperature": 0.0,
        "max_tokens": 120
    }

    parsed_json = None
    connected_url = None

    urls = [] if offline else ([ollama_url] if ollama_url else list(dict.fromkeys(candidate_urls)))
    for url in urls:
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            raw_content = res["choices"][0]["message"]["content"].strip()

            # Pulizia tag o markdown
            cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            parsed_json = json.loads(cleaned)
            connected_url = url
            break
        except Exception:
            continue

    if parsed_json is None:
        print("Parser deterministico richiesto." if offline else
              "Avviso: Server Ollama non direttamente raggiungibile, applicazione del parser clinico deterministico.")
        instr_lower = instruction_text.lower()
        if "bisturi" in instr_lower or "scalpel" in instr_lower:
            tool = "scalpel"
            kw = "scalpel"
        elif "pinza" in instr_lower or "forceps" in instr_lower:
            tool = "hemostatic forceps"
            kw = "forceps"
        else:
            tool = "vascular scissors"
            kw = "scissors"

        src = "tray_2" if "2" in instr_lower else "tray_1"
        dst = "tray_1" if src == "tray_2" else "tray_2"

        parsed_json = {
            "action": "pick_and_place",
            "target_instrument": tool,
            "target_keyword": kw,
            "source_tray": src,
            "destination_tray": dst
        }
    else:
        print(f"Risposta LLM ricevuta da: {connected_url}")

    # Costruzione del report strutturato per la catena
    raw_tool = target_tool_hint or parsed_json.get("target_instrument") or "vascular scissors"
    raw_tool = str(raw_tool).strip()

    raw_kw = str(parsed_json.get("target_keyword", "")).strip().lower()
    # Se il keyword è vuoto o incongruente con il nome dello strumento
    if not raw_kw or (raw_kw == "scissors" and "scissor" not in raw_tool.lower()):
        raw_kw = extract_keyword_from_instrument(raw_tool)

    config = load_environment_config(config_path)
    report = {
        "stage": "stage1_task_plan",
        "timestamp": datetime.datetime.now().isoformat(),
        "instruction": instruction_text,
        "action": normalize_action(parsed_json.get("action")),
        "target_instrument": raw_tool,
        "target_keyword": raw_kw,
        "source_tray": normalize_tray(parsed_json.get("source_tray"), default="tray_2", config=config),
        "destination_tray": normalize_tray(parsed_json.get("destination_tray"), default="tray_1", config=config),
        "planner_backend": "ollama" if connected_url else "deterministic_parser",
        "status": "COMPLETED"
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n[STAGE 1] REPORT JSON GENERATO:")
    print(json.dumps(report, indent=2))
    print(f"Salvato in: {output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: LLM Task Planner")
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION,
                        help="Comando in linguaggio naturale dell'operatore")
    parser.add_argument("--output", type=str, default=None,
                        help="Percorso file output stage1_task_plan.json")
    args = parser.parse_args()

    run_llm_planner(instruction_text=args.instruction, output_path=args.output)
