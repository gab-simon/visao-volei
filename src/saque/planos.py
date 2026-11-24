"""Parte 2: trecho de câmera aberta em volta do contato e homografias frame → contato.

Andamos do contato para trás e para frente enquanto o frame registra no vizinho e,
alinhado, continua parecido com o frame do contato (isso barra cortes e vinhetas).

    uv run saque-planos [--saque rally_000140_serve0]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from .dados import ALTURA, FPS, LARGURA, PLACAR, PROC, Saque, salvar_json, saques_validos
from .registro import features, homografia

DIR_H = PROC / "homografias"
DIR_VISUAL = PROC / "planos_visual"
NCC_MIN = 0.80   # plano aberto fica acima de 0,81; vinhetas abaixo de 0,75
ESCALA = 0.5


def ncc_alinhado(img: np.ndarray, ref: np.ndarray, H: np.ndarray) -> float:
    cinza = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
    alinhado = cv2.warpPerspective(cinza(img), H, (LARGURA, ALTURA))
    valido = cv2.warpPerspective(np.ones((ALTURA, LARGURA), np.float32), H, (LARGURA, ALTURA))
    valido[PLACAR] = 0
    peq = lambda x: cv2.resize(x, (160, 90), interpolation=cv2.INTER_AREA)
    m = peq(valido) > 0.99
    a, b = peq(alinhado)[m], peq(cinza(ref))[m]
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6))


def plano_aberto(s: Saque) -> dict[int, np.ndarray]:
    """{k: H_k} para os frames abertos; H_k leva pixels do frame k para o frame do contato."""
    ref = s.ler(s.frame)
    H = {s.frame: np.eye(3)}
    for passo in (-1, 1):
        k, f_k = s.frame, features(ref, ESCALA)
        while k + passo in s.janela:
            img = s.ler(k + passo)
            f_viz = features(img, ESCALA)
            H_viz = homografia(f_viz, f_k)
            if H_viz is None:
                break
            H_viz = H[k] @ H_viz
            if ncc_alinhado(img, ref, H_viz) < NCC_MIN:
                break
            H[k + passo] = H_viz / H_viz[2, 2]
            k, f_k = k + passo, f_viz
    return H


def tira_visual(s: Saque, ini: int, fim: int) -> np.ndarray:
    """Miniaturas a cada 0,4 s: borda verde no plano aberto, vermelha fora."""
    tiles = []
    for k in s.janela[::10]:
        img = cv2.resize(s.ler(k), (256, 144))
        cor = (0, 200, 0) if ini <= k <= fim else (0, 0, 220)
        img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=cor)
        cv2.putText(img, f"{(k - s.frame) / FPS:+.1f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        tiles.append(img)
    return np.hstack(tiles)


def processar(s: Saque) -> dict:
    H = plano_aberto(s)
    ks = sorted(H)
    np.savez_compressed(DIR_H / f"{s.serve_id}.npz", frames=ks, H=[H[k] for k in ks])
    cv2.imwrite(str(DIR_VISUAL / f"{s.serve_id}.jpg"), tira_visual(s, ks[0], ks[-1]))
    return {"aberto": [ks[0], ks[-1]], "segundos_antes": round((s.frame - ks[0]) / FPS, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saque")
    args = ap.parse_args()
    DIR_H.mkdir(parents=True, exist_ok=True)
    DIR_VISUAL.mkdir(parents=True, exist_ok=True)

    saques = [s for s in saques_validos() if args.saque in (None, s.serve_id)]
    with ProcessPoolExecutor(8) as ex:
        resumo = dict(zip((s.serve_id for s in saques), ex.map(processar, saques)))
    for sid, r in resumo.items():
        print(f"{sid}: aberto {r['aberto']}, {r['segundos_antes']:.2f}s antes do contato")
    if args.saque is None:
        salvar_json(PROC / "planos.json", resumo)
        seg = np.array([r["segundos_antes"] for r in resumo.values()])
        print(f"\nmediana de plano aberto antes do contato: {np.median(seg):.2f}s | "
              f"≥0,4s: {(seg >= 0.4).sum()} | ≥1s: {(seg >= 1).sum()}")


if __name__ == "__main__":
    main()
