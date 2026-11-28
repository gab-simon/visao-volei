"""Partes 4 e 5: bola no lançamento e altura máxima.

4) Em cada frame aberto antes do contato, alinha os frames k−2 e k+2 no frame k pelas
   homografias da parte 2 e fica com o que mudou nos dois (diferença tripla). As manchas
   pequenas e compactas perto do sacador são candidatas a bola.
5) A bola em queda livre segue X(t) = X₀ + V₀·t + ½·g·t². Um RANSAC sorteia 3 candidatos
   de frames diferentes, resolve X₀ e V₀ (sistema linear, 6 × 6) e conta quantos frames
   têm um candidato perto da parábola projetada. As melhores são refinadas por mínimos
   quadrados no erro de reprojeção, e fica a plausível que explica mais frames. A altura máxima sai da parábola, mesmo sem ver o topo.

Tudo é feito nas coordenadas do frame do contato (pixels levados por H_k), projetando
com a P do contato. t = 0 no contato.

    uv run saque-bola [--saque rally_000140_serve0]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.optimize import least_squares

from .calibracao import desenhar, projetar
from .dados import ALTURA, FPS, LARGURA, PLACAR, PROC, Saque, ler_json, salvar_json, saques_validos

DIR_BOLA = PROC / "bola"
DIR_VISUAL = PROC / "bola_visual"
G = np.array([0.0, 0.0, -9.81])

PASSO = 2                    # vizinhos da diferença tripla: k−2 e k+2
LIMIAR_DIF = 22              # níveis de cinza
AREA = (4, 220)              # px² da mancha
REGIAO = (450, 550, 200)     # px em volta do contato: para os lados, para cima, para baixo
DESVIO_PX = 3.0              # trecho de 3 frames: o do meio a até 3 px do ponto médio dos vizinhos
PASSO_PX = (1.5, 50)         # deslocamento por frame
MIN_FRAMES = 7               # frames abertos antes do contato para tentar o ajuste
ITERACOES = 20000
INLIER_PX = 4.0
MIN_INLIERS = 7              # frames com bola na parábola
REFINAR = 100                # hipóteses do RANSAC refinadas


# ---------------------------------------------------------------- 4) candidatos

def _cinza(s: Saque, k: int) -> np.ndarray:
    return cv2.cvtColor(s.ler(k), cv2.COLOR_BGR2GRAY).astype(np.float32)


def candidatos(s: Saque, H: dict[int, np.ndarray]) -> list[tuple[int, float, float]]:
    """(k, u, v) das manchas em movimento, já em pixels do frame do contato."""
    u0, v0 = s.xy[0] * LARGURA, s.xy[1] * ALTURA
    lado, cima, baixo = REGIAO
    saida = []
    for k in sorted(k for k in H if k < s.frame):
        vizinhos = [j for j in (k - PASSO, k + PASSO) if j in H] or [j for j in (k - 1, k + 1) if j in H]
        if len(vizinhos) < 2:
            continue
        Ik = _cinza(s, k)
        dif = np.minimum(*[np.abs(Ik - cv2.warpPerspective(_cinza(s, j), np.linalg.inv(H[k]) @ H[j],
                                                              (LARGURA, ALTURA))) for j in vizinhos])
        m = (dif > LIMIAR_DIF).astype(np.uint8)
        m[PLACAR] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, _, stats, centros = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not AREA[0] <= area <= AREA[1] or max(w, h) > 4 * min(w, h) or area < 0.35 * w * h:
                continue
            u, v = cv2.perspectiveTransform(centros[i][None, None].astype(np.float64), H[k])[0, 0]
            if abs(u - u0) < lado and v0 - cima < v < v0 + baixo:
                saida.append((k, float(u), float(v)))
    return em_movimento(saida)


def em_movimento(cands: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    """Só candidatos que formam, com um vizinho em k−1 e outro em k+1, um trecho quase reto
    e uniforme que de fato se desloca. Manchas paradas e ruído solto ficam de fora."""
    por_frame: dict[int, np.ndarray] = {}
    for k, u, v in cands:
        por_frame.setdefault(k, []).append((u, v))
    por_frame = {k: np.array(p) for k, p in por_frame.items()}
    manter = set()
    for k, atual in por_frame.items():
        if k - 1 not in por_frame or k + 1 not in por_frame:
            continue
        antes, depois = por_frame[k - 1], por_frame[k + 1]
        meio = (antes[:, None] + depois[None]) / 2                       # (a, d, 2)
        passo = np.linalg.norm(depois[None] - antes[:, None], axis=-1) / 2
        for i, p in enumerate(atual):
            a, d = np.nonzero((np.linalg.norm(meio - p, axis=-1) < DESVIO_PX)
                              & (passo > PASSO_PX[0]) & (passo < PASSO_PX[1]))
            if len(a):
                manter |= {(k, *p)} | {(k - 1, *antes[j]) for j in a} | {(k + 1, *depois[j]) for j in d}
    return sorted((int(k), float(u), float(v)) for k, u, v in manter)


# ---------------------------------------------------------------- 5) parábola

def posicoes(X0: np.ndarray, V0: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)[..., None]
    return X0 + V0 * t + 0.5 * G * t ** 2


def _linhas(P: np.ndarray, u: np.ndarray, v: np.ndarray, t: np.ndarray):
    """Duas equações lineares em (X₀, V₀) por observação: (u·P₃ − P₁)·[X(t); 1] = 0, idem v."""
    A, b = [], []
    for a in (u[:, None] * P[2] - P[0], v[:, None] * P[2] - P[1]):
        A.append(np.concatenate([a[:, :3], a[:, :3] * t[:, None]], axis=1))
        b.append(-(a[:, :3] @ (0.5 * G) * t ** 2 + a[:, 3]))
    return np.stack(A, 1), np.stack(b, 1)           # (n, 2, 6), (n, 2)


def _plausivel(X0: np.ndarray, V0: np.ndarray) -> np.ndarray:
    """No contato: atrás ou perto da linha de fundo, entre 1,8 e 4,5 m, bola não subindo rápido."""
    x, y, z = X0[:, 0], X0[:, 1], X0[:, 2]
    return ((np.abs(x) > 7) & (np.abs(x) < 20) & (np.abs(y) < 6) & (z > 1.8) & (z < 4.5)
            & (V0[:, 2] < 1.5) & (np.linalg.norm(V0, axis=1) < 15))


def _erros_por_frame(P, X0, V0, k, uv, t, frames):
    """Para cada hipótese, a menor distância (px) entre a parábola e os candidatos de cada frame."""
    X = posicoes(X0[:, None], V0[:, None], t)                       # (N, M, 3)
    x = np.concatenate([X, np.ones(X.shape[:2] + (1,))], -1) @ P.T
    with np.errstate(divide="ignore", invalid="ignore"):
        erro = np.linalg.norm(x[..., :2] / x[..., 2:] - uv, axis=-1)
    erro[~np.isfinite(erro) | (x[..., 2] <= 0)] = np.inf
    inicio = np.searchsorted(k, frames)
    return np.minimum.reduceat(erro, inicio, axis=1)                # (N, frames)


def ajustar(P: np.ndarray, cands: list[tuple[int, float, float]], frame_contato: int,
            rng: np.random.Generator) -> dict | None:
    if not cands:
        return None
    c = np.array(sorted(cands))
    k, uv = c[:, 0].astype(int), c[:, 1:]
    t = (k - frame_contato) / FPS
    frames = np.unique(k)
    if len(frames) < MIN_INLIERS:
        return None
    A, b = _linhas(P, uv[:, 0], uv[:, 1], t)

    # sorteia 3 candidatos de frames distintos
    idx = rng.integers(0, len(c), size=(ITERACOES, 3))
    distintos = (k[idx[:, 0]] != k[idx[:, 1]]) & (k[idx[:, 0]] != k[idx[:, 2]]) & (k[idx[:, 1]] != k[idx[:, 2]])
    idx = np.unique(np.sort(idx[distintos], axis=1), axis=0)
    if len(idx) == 0:
        return None
    As, bs = A[idx].reshape(-1, 6, 6), b[idx].reshape(-1, 6)
    ok = np.abs(np.linalg.det(As)) > 1e-9
    sol = np.linalg.solve(As[ok], bs[ok][..., None])[..., 0]
    # sem filtro aqui: com 3 pontos a profundidade é instável e boas hipóteses parecem absurdas;
    # a plausibilidade só é cobrada depois do refino
    X0, V0 = sol[:, :3], sol[:, 3:]

    # as melhores hipóteses, refinadas; fica a que explica mais frames
    notas = []
    for lote in range(0, len(X0), 2000):
        e = _erros_por_frame(P, X0[lote:lote + 2000], V0[lote:lote + 2000], k, uv, t, frames)
        dentro = e < INLIER_PX
        notas.append(dentro.sum(1) - np.where(dentro, e, 0).sum(1) / (INLIER_PX * len(frames) + 1))
    notas = np.concatenate(notas)
    melhores = [r for i in np.argsort(-notas)[:REFINAR]
                if (r := _refinar(P, X0[i], V0[i], k, uv, t, frames)) is not None]
    return max(melhores, key=lambda r: (len(r["k"]), -r["rms_px"]), default=None)


def _refinar(P, x0, v0, k, uv, t, frames) -> dict | None:
    """Em cada frame, o candidato mais perto da parábola; reajusta e repete."""
    for _ in range(3):
        dist = np.linalg.norm(projetar(P, posicoes(x0, v0, t)) - uv, axis=1)
        sel = np.array([i for f in frames for i in [np.flatnonzero(k == f)[np.argmin(dist[k == f])]]
                        if dist[i] < 2 * INLIER_PX])
        if len(sel) < MIN_INLIERS:
            return None

        def residuos(p):
            return (projetar(P, posicoes(p[:3], p[3:], t[sel])) - uv[sel]).ravel()

        x0, v0 = np.split(least_squares(residuos, np.r_[x0, v0], loss="huber", f_scale=2.0).x, 2)

    e = np.linalg.norm(projetar(P, posicoes(x0, v0, t[sel])) - uv[sel], axis=1)
    sel, e = sel[e < INLIER_PX], e[e < INLIER_PX]
    if len(sel) < MIN_INLIERS or not _plausivel(x0[None], v0[None])[0]:
        return None
    t_pico = v0[2] / -G[2]  # vz(t) = vz0 + g·t = 0
    return {"X0": x0, "V0": v0, "k": k[sel], "uv": uv[sel], "rms_px": float(np.sqrt(np.mean(e ** 2))),
            "altura_max_m": float(posicoes(x0, v0, t_pico)[2]), "t_pico_s": float(t_pico)}


# ---------------------------------------------------------------- uso e visualização

def visual(s: Saque, P: np.ndarray, cands, r: dict | None) -> np.ndarray:
    img = desenhar(s.ler(s.frame), P)
    for _, u, v in cands:
        cv2.circle(img, (round(u), round(v)), 3, (0, 0, 255), 1, cv2.LINE_AA)
    texto = f"{s.serve_id}  sem ajuste"
    if r is not None:
        ts = np.linspace(min(r["t_pico_s"], (r["k"].min() - s.frame) / FPS) - 0.1, 0, 60)
        pts = projetar(P, posicoes(r["X0"], r["V0"], ts))
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, (0, 255, 255), 1, cv2.LINE_AA)
        for u, v in r["uv"]:
            cv2.circle(img, (round(u), round(v)), 5, (0, 255, 0), 1, cv2.LINE_AA)
        texto = (f"{s.serve_id}  altura max {r['altura_max_m']:.2f} m  contato {r['X0'][2]:.2f} m  "
                 f"{len(r['k'])} frames  rms {r['rms_px']:.2f}px")
    for cor, esp in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(img, texto, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, esp)
    u0, v0 = s.xy[0] * LARGURA, s.xy[1] * ALTURA
    lado, cima, baixo = REGIAO
    y0, x0 = max(0, int(v0 - cima)), max(0, int(u0 - lado))
    return img[y0:int(v0 + baixo), x0:int(u0 + lado)]


def processar(s: Saque) -> dict:
    z = np.load(PROC / "homografias" / f"{s.serve_id}.npz")
    H = dict(zip(z["frames"].tolist(), z["H"]))
    P = np.load(PROC / "calibracao" / f"{s.serve_id}.npz")["P"]
    cands = candidatos(s, H)
    r = ajustar(P, cands, s.frame, np.random.default_rng(0))
    cv2.imwrite(str(DIR_VISUAL / f"{s.serve_id}.jpg"), visual(s, P, cands, r))
    if r is None:
        return {"ok": False, "candidatos": len(cands)}
    np.savez(DIR_BOLA / f"{s.serve_id}.npz", X0=r["X0"], V0=r["V0"], frames=r["k"], uv=r["uv"])
    pico_visto = r["k"].min() <= s.frame + r["t_pico_s"] * FPS
    return {"ok": True, "candidatos": len(cands), "frames": len(r["k"]), "rms_px": round(r["rms_px"], 2),
            "altura_max_m": round(r["altura_max_m"], 2), "altura_contato_m": round(float(r["X0"][2]), 2),
            "t_pico_s": round(r["t_pico_s"], 2), "pico_visto": bool(pico_visto),
            "contato_xyz_m": np.round(r["X0"], 2).tolist(), "v_contato_ms": np.round(r["V0"], 2).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saque")
    args = ap.parse_args()
    DIR_BOLA.mkdir(parents=True, exist_ok=True)
    DIR_VISUAL.mkdir(parents=True, exist_ok=True)

    calibrados = {sid for sid, v in ler_json(PROC / "calibracao.json")["saques"].items() if v["ok"]}
    abertos = ler_json(PROC / "planos.json")
    saques = [s for s in saques_validos() if args.saque in (None, s.serve_id) and s.serve_id in calibrados
              and s.frame - abertos[s.serve_id]["aberto"][0] >= MIN_FRAMES]
    with ProcessPoolExecutor(8) as ex:
        resumo = dict(zip((s.serve_id for s in saques), ex.map(processar, saques)))
    for sid, r in resumo.items():
        if r["ok"]:
            print(f"{sid}: {r['altura_max_m']:.2f} m (contato {r['altura_contato_m']:.2f} m), "
                  f"{r['frames']} frames, rms {r['rms_px']:.2f}px{'' if r['pico_visto'] else ', pico extrapolado'}")
        else:
            print(f"{sid}: sem ajuste ({r['candidatos']} candidatos)")
    if args.saque is None:
        salvar_json(PROC / "bola.json", resumo)
        alt = np.array([r["altura_max_m"] for r in resumo.values() if r["ok"]])
        print(f"\n{len(alt)}/{len(saques)} saques com parábola | altura máxima mediana {np.median(alt):.2f} m "
              f"(p10 {np.percentile(alt, 10):.2f}, p90 {np.percentile(alt, 90):.2f})")


if __name__ == "__main__":
    main()
