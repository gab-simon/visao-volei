"""GIF em câmera lenta de um saque, do começo do plano aberto até o contato, com a
trajetória calculada (parte 5) sobre o vídeo.

Em cada frame k a parábola é projetada com P_k = H_k⁻¹ · P_contato, então ela acompanha
a câmera quando ela gira ou dá zoom. Amarelo: parábola inteira; branco: trecho já
percorrido; verde: bola detectada no frame; linha tracejada: altura máxima até o chão.

    uv run saque-gif [--saque rally_000026_serve0] [--fps 6] [--saida data/gifs]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .bola import G, posicoes
from .calibracao import P_do_frame, projetar
from .dados import DATA, FPS, PROC, ler_json, saque

RECORTE = (800, 450)   # px do frame original em volta da trajetória
SAIDA_LARGURA = 800


def _texto(img: np.ndarray, texto: str, xy: tuple[int, int], escala: float = 0.5, cor=(255, 255, 255)) -> None:
    """Texto com contorno fino (o OpenCV só desenha ASCII)."""
    for c, esp in (((0, 0, 0), 3), (cor, 1)):
        cv2.putText(img, texto, xy, cv2.FONT_HERSHEY_SIMPLEX, escala, c, esp, cv2.LINE_AA)


def _faixa(img: np.ndarray, texto: str, y: int) -> None:
    """Texto branco sobre uma faixa escura translúcida, de ponta a ponta."""
    y0, y1 = max(y - 20, 0), min(y + 8, img.shape[0])
    img[y0:y1] = (img[y0:y1] * 0.35).astype(np.uint8)
    cv2.putText(img, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _tracejada(img, a, b, cor, passo: int = 8) -> None:
    a, b = np.array(a, float), np.array(b, float)
    n = max(int(np.linalg.norm(b - a) / passo), 1)
    for i in range(0, n, 2):
        p, q = a + (b - a) * i / n, a + (b - a) * min(i + 1, n) / n
        cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), cor, 1, cv2.LINE_AA)


def quadro(s, k: int, X0, V0, t_ini: float, deteccoes: dict, info: dict) -> np.ndarray:
    P = P_do_frame(s.serve_id, k)
    img = s.ler(k)
    t_k = (k - s.frame) / FPS
    ts = np.linspace(t_ini, 0, 120)
    pts = projetar(P, posicoes(X0, V0, ts))
    cv2.polylines(img, [np.round(pts).astype(np.int32)], False, (0, 220, 255), 1, cv2.LINE_AA)
    if t_k > t_ini:
        feito = pts[ts <= min(t_k, 0)]
        if len(feito) > 1:
            cv2.polylines(img, [np.round(feito).astype(np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)

    # altura máxima: do topo até o chão logo abaixo
    t_pico = V0[2] / -G[2]
    topo = posicoes(X0, V0, t_pico)
    a, b = projetar(P, np.array([topo, [topo[0], topo[1], 0.0]]))
    _tracejada(img, a, b, (0, 220, 255))
    _texto(img, f"max {topo[2]:.2f} m", (int(a[0]) + 8, int(a[1]) - 8), 0.5, (0, 220, 255))

    if t_ini <= t_k <= 0:
        bola = projetar(P, posicoes(X0, V0, t_k)[None])[0]
        cv2.circle(img, tuple(np.round(bola).astype(int)), 9, (255, 255, 255), 1, cv2.LINE_AA)
        _texto(img, f"{posicoes(X0, V0, t_k)[2]:.2f} m", (int(bola[0]) + 12, int(bola[1]) + 14))
    if k in deteccoes:
        # detecções estão no frame do contato; H_k⁻¹ leva de volta ao frame k
        u, v = cv2.perspectiveTransform(np.array([[deteccoes[k]]], float), np.linalg.inv(info["H"][k]))[0, 0]
        cv2.circle(img, (round(u), round(v)), 5, (0, 255, 0), 1, cv2.LINE_AA)

    # recorte em volta da trajetória deste frame
    cx, cy = np.mean([pts.min(0), pts.max(0)], axis=0)
    w, h = RECORTE
    x0 = int(np.clip(cx - w / 2, 0, img.shape[1] - w))
    y0 = int(np.clip(cy - h / 2 + 60, 0, img.shape[0] - h))
    img = img[y0:y0 + h, x0:x0 + w]
    img = cv2.resize(img, (SAIDA_LARGURA, round(SAIDA_LARGURA * h / w)), interpolation=cv2.INTER_AREA)

    fase = "lancamento" if t_k < 0 else "contato"
    _faixa(img, f"{s.serve_id}    t = {t_k:+.2f} s    {fase}", 22)
    _faixa(img, f"altura max {topo[2]:.2f} m    contato {X0[2]:.2f} m    ajuste: {info['n']} frames, "
                f"rms {info['rms']:.2f} px", img.shape[0] - 10)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saque", default="rally_000026_serve0")
    ap.add_argument("--fps", type=float, default=6, help="quadros por segundo do GIF (o vídeo tem 25)")
    ap.add_argument("--saida", default=str(DATA / "gifs"))
    args = ap.parse_args()

    s = saque(args.saque)
    resumo = ler_json(PROC / "bola.json").get(s.serve_id)
    if not resumo or not resumo["ok"]:
        raise SystemExit(f"{s.serve_id} não tem trajetória (rode antes: uv run saque-bola)")
    z = np.load(PROC / "bola" / f"{s.serve_id}.npz")
    hz = np.load(PROC / "homografias" / f"{s.serve_id}.npz")
    H = dict(zip(hz["frames"].tolist(), hz["H"]))
    deteccoes = dict(zip(z["frames"].tolist(), z["uv"].tolist()))
    t_ini = (min(deteccoes) - s.frame) / FPS
    info = {"H": H, "n": len(deteccoes), "rms": resumo["rms_px"]}

    ks = [k for k in sorted(H) if k <= s.frame]     # termina no contato
    quadros = [quadro(s, k, z["X0"], z["V0"], t_ini, deteccoes, info) for k in ks]

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    arquivo = saida / f"{s.serve_id}.gif"
    imgs = [Image.fromarray(cv2.cvtColor(q, cv2.COLOR_BGR2RGB)) for q in quadros]
    duracao = [round(1000 / args.fps)] * len(imgs)
    duracao[-1] = 1500                                   # segura no contato
    imgs[0].save(arquivo, save_all=True, append_images=imgs[1:], duration=duracao, loop=0, optimize=True)
    print(f"{arquivo}: {len(imgs)} frames ({ks[0] - s.frame:+d} a {ks[-1] - s.frame:+d}), "
          f"{25 / args.fps:.1f}× mais lento, {arquivo.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
