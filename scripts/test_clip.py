#!/usr/bin/env python3
"""Prova offline della pipeline RGB-D su uno o più frame, senza movimento robot."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clip', type=Path, default=ROOT / 'execution/clip')
    p.add_argument('--indices', type=int, nargs='+', default=[0], help='Indici, per esempio 0 10 20')
    p.add_argument('--crop', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                   help='Ritaglio RGB-D in pixel; le intrinseche vengono traslate automaticamente')
    p.add_argument('--depth-scale', type=float, required=True, help='Metri per unità del PNG; verificare alla sorgente')
    p.add_argument('--out-dir', type=Path, default=ROOT / 'data/outputs/clip_test')
    p.add_argument('--instruction', default='Prendi le forbici dal vassoio 2 e mettile nel vassoio 1')
    p.add_argument('--expected-keyword', required=True, help='Bersaglio atteso in inglese, per esempio scissors')
    p.add_argument('--sam3-model', default=str(ROOT / 'data/external/sam3-prepared'))
    p.add_argument('--sam3-device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--stop-after', choices=['capture', 'segment', 'deproject', 'grasp'], default='grasp')
    p.add_argument('--graspgen-host', default='127.0.0.1')
    p.add_argument('--graspgen-port', type=int, default=5556)
    args = p.parse_args()
    if not math.isfinite(args.depth_scale) or args.depth_scale <= 0:
        p.error('--depth-scale deve essere positivo e finito')
    if len(set(args.indices)) != len(args.indices) or any(i < 0 for i in args.indices):
        p.error('gli indici devono essere non negativi e distinti')
    clip = args.clip.resolve()
    meta = yaml.safe_load((clip / 'camera.yaml').read_text())
    if meta.get('depth_aligned_to_color') is not True:
        p.error('camera.yaml non dichiara depth_aligned_to_color: true')
    k = meta['intrinsics_color']
    intr = [float(k[x]) for x in ('fu', 'fv', 'ppu', 'ppv')]
    if not all(math.isfinite(x) for x in intr) or min(intr[:2]) <= 0:
        p.error('intrinseche non valide')
    print('ATTENZIONE: scala fornita dall’utente; allineamento dichiarato ma non verificato; '
          'distorsione non corretta e posa camera nominale. Nessun movimento.', flush=True)
    for index in args.indices:
        image = clip / f'color_{index:06d}.png'
        depth = clip / f'depth_{index:06d}.png'
        if not image.is_file() or not depth.is_file():
            p.error(f'coppia mancante per indice {index}: {image}, {depth}')
        with Image.open(image) as im:
            size = im.size
        with Image.open(depth) as im:
            raw = np.array(im)
        if size != (int(k['width']), int(k['height'])) or raw.shape != (size[1], size[0]) or raw.dtype != np.uint16:
            p.error(f'frame {index}: dimensioni o formato profondità incompatibili con camera.yaml')
        out = (args.out_dir / f'frame_{index:06d}').resolve()
        if out.exists() and any(out.iterdir()):
            p.error(f'output già presente: {out}; scegliere un’altra directory per evitare risultati vecchi')
        out.mkdir(parents=True, exist_ok=True)
        image_input = image
        adjusted_intr = intr.copy()
        if args.crop:
            x, y, w, h = args.crop
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > size[0] or y + h > size[1]:
                p.error(f'ritaglio fuori dall’immagine {size}: {args.crop}')
            image_input = out / 'color_crop.png'
            with Image.open(image) as im:
                im.crop((x, y, x + w, y + h)).save(image_input)
            raw = raw[y:y + h, x:x + w]
            adjusted_intr[2] -= x
            adjusted_intr[3] -= y
        np.save(out / 'depth_metric_m.npy', raw.astype(np.float32) * args.depth_scale)
        (out / 'clip_input.json').write_text(json.dumps({'index': index, 'camera_yaml': str(clip / 'camera.yaml'),
            'color': str(image), 'depth': str(depth), 'depth_scale_m_per_unit_assumed': args.depth_scale,
            'depth_aligned_to_color_declared': True, 'distortion_corrected': False,
            'camera_pose_verified': False, 'crop_xywh': args.crop,
            'intrinsics_used_fx_fy_cx_cy': adjusted_intr}, indent=2) + '\n')
        cmd = [sys.executable, str(ROOT / 'run_pipeline_real.py'), '--image', str(image_input),
               '--depth', str(out / 'depth_metric_m.npy'), '--intrinsics', *map(str, adjusted_intr),
               '--instruction', args.instruction, '--out-dir', str(out), '--sam3-model', args.sam3_model,
               '--sam3-device', args.sam3_device, '--grasp-backend', 'graspgen',
               '--graspgen-host', args.graspgen_host, '--graspgen-port', str(args.graspgen_port)]
        print('Frame', index, '→', out, flush=True)
        result = subprocess.run(cmd + ['--stop-after', 'plan'], cwd=ROOT)
        if result.returncode:
            return result.returncode
        plan = json.loads((out / 'plan.json').read_text())['data']
        if plan.get('planner_backend') != 'ollama' or plan.get('target_keyword', '').lower() != args.expected_keyword.lower():
            print(f"Piano rifiutato: backend={plan.get('planner_backend')!r}, bersaglio={plan.get('target_keyword')!r}; "
                  f"attesi Ollama e {args.expected_keyword!r}. Controllare istruzione e LLM.", file=sys.stderr)
            return 1
        result = subprocess.run(cmd + ['--from-stage', 'capture', '--stop-after', args.stop_after], cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
