"""
Modulo 1: Task Planning Naturale con LLM (Qwen 2.5 3B via Ollama).
Traduce l'istruzione del chirurgo/operatore in un piano d'azione JSON standardizzato.
"""

import os
import json
import re
import urllib.request
import shutil

def plan_surgical_task(instruction_text, ollama_url="http://172.19.128.1:11434/v1/chat/completions", model="qwen2.5:3b"):
    print("\n" + "="*65)
    print("[FASE 1/4] LLM TASK PLANNER (Qwen 2.5 su Ollama)")
    print(f"   Istruzione ricevuta: \"{instruction_text}\"")
    print("="*65)

    system_prompt = (
        "You are an AI medical robot task planner. "
        "Your job is to parse the user's natural language instruction and output a JSON object with EXACTLY three fields:\n"
        "  1. 'action': the high-level robotic operation, typically 'pick_and_place' or 'pick' or 'place'.\n"
        "  2. 'target_prompt': the English noun phrase identifying the requested medical instrument (e.g. 'surgical scissors', 'scalpel', 'forceps').\n"
        "  3. 'destination': the target location or receptacle (e.g. 'tray_1', 'tray_2', 'table').\n\n"
        "Example input: 'Pick up the surgical scissors and place them in tray 1'\n"
        "Example output:\n"
        "{\n"
        "  \"action\": \"pick_and_place\",\n"
        "  \"target_prompt\": \"surgical scissors\",\n"
        "  \"destination\": \"tray_1\"\n"
        "}\n\n"
        "Respond ONLY with the valid JSON object, without markdown formatting or other text."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction_text}
        ],
        "temperature": 0.0,
        "max_tokens": 80
    }

    try:
        req = urllib.request.Request(
            ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        raw_content = res["choices"][0]["message"]["content"].strip()

        # Pulizia tag o blocchi markdown
        cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        task_data = json.loads(cleaned)
    except Exception as e:
        print(f"Avviso: Errore connessione Ollama ({e}), utilizzo piano clinico standard.")
        dest = "tray_2" if "2" in instruction_text else "tray_1"
        tool = "scalpel" if "bisturi" in instruction_text.lower() else "surgical scissors"
        task_data = {
            "action": "pick_and_place",
            "target_prompt": tool,
            "destination": dest
        }

    print("\nTask Plan JSON Generato:")
    print(json.dumps(task_data, indent=2))

    # Salvataggio organizzato in data/outputs/
    out_dir = "/workspace/data/outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "task_plan.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(task_data, f, indent=2)

    print(f"File generato salvato in: {out_path}")
    return task_data
