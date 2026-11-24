"""Parte 1: extrai do vídeo completo os frames de −3 s a +1 s em volta de cada saque.

    uv run saque-janelas [--saque rally_000140_serve0] [--refazer]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .dados import DATA, FPS, FRAMES_720, VIDEO, Saque, saques_validos


def extrair(s: Saque, refazer: bool = False) -> None:
    pasta = s.caminho(0).parent
    if not refazer and len(list(pasta.glob("*.jpg"))) == len(s.janela):
        return
    # com -ss no instante exato o ffmpeg entrega o frame anterior; meio frame resolve
    inicio = s.tempo(s.janela[0]) + 0.5 / FPS
    with tempfile.TemporaryDirectory(dir=DATA) as tmp:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{inicio:.3f}", "-i", str(VIDEO),
                        "-frames:v", str(len(s.janela)), "-q:v", "2", f"{tmp}/%04d.jpg"], check=True)
        shutil.rmtree(pasta, ignore_errors=True)
        pasta.mkdir(parents=True)
        for k, arquivo in zip(s.janela, sorted(Path(tmp).glob("*.jpg"))):
            arquivo.rename(s.caminho(k))


def _cinza(caminho: Path) -> np.ndarray:
    img = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)


def no_frame_certo(s: Saque) -> bool:
    """O frame extraído k tem de ser mais parecido com o frame k do 720p do que com k±1."""
    ref = lambda k: FRAMES_720 / s.rally / f"{k:06d}.jpg"
    comuns = [k for k in s.janela if all(ref(k + d).exists() for d in (-1, 0, 1))]
    votos = []
    for k in comuns[:: max(1, len(comuns) // 6)][:6]:
        novo = _cinza(s.caminho(k))
        erro = [np.abs(novo - _cinza(ref(k + d))).mean() for d in (-1, 0, 1)]
        votos.append(int(np.argmin(erro)) == 1)
    return sum(votos) > len(votos) / 2   # maioria: vinhetas enganam um frame ou outro


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saque")
    ap.add_argument("--refazer", action="store_true")
    args = ap.parse_args()
    if not VIDEO.exists():
        raise SystemExit(f"vídeo não encontrado: {VIDEO} (defina SAQUE_VIDEO)")

    saques = [s for s in saques_validos() if args.saque in (None, s.serve_id)]

    def tarefa(s: Saque) -> bool:
        extrair(s, args.refazer)
        return no_frame_certo(s)

    with ThreadPoolExecutor(6) as ex:
        certos = list(ex.map(tarefa, saques))
    for s, ok in zip(saques, certos):
        if not ok:
            print(f"fora de sincronia: {s.serve_id}")
    print(f"{len(saques)} saques extraídos, {sum(certos)} no frame exato")


if __name__ == "__main__":
    main()
