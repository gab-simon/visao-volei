"""Parte 3: calibração da câmera aberta.

A) Em cada saque, ajusta as 6 linhas da quadra nos pixels brancos, partindo dos pontos
   marcados à mão no frame de referência e levados até o saque por homografia.
B) Com todos os saques juntos, estima uma posição única da câmera (ela não sai do lugar)
   e, por saque, rotação e focal. Hipóteses: pixel quadrado, centro óptico no meio da
   imagem, sem distorção de lente.

    uv run saque-calibracao
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .dados import ALTURA, ANN, LARGURA, PLACAR, PROC, Saque, ler_json, salvar_json, saque, saques_validos
from .quadra import LINHAS, PONTOS, cruzamento, ponto, reta, segmentos_3d
from .registro import features, homografia

DIR_CAL = PROC / "calibracao"
DIR_VISUAL = PROC / "calibracao_visual"
FAIXAS = (24, 12, 6, 4)                  # px em volta de cada linha, estreitando a cada passo
C_INICIAL = np.array([0.0, -30.0, 10.0])  # arquibancada lateral, alinhada com a rede
RMS_MAX = 3.0
CX, CY = LARGURA / 2, ALTURA / 2


# ---------------------------------------------------------------- A) linhas da quadra

def mascara_linhas(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    m = (tophat > 14) & (hsv[..., 1] < 90) & (v > 120)
    m[PLACAR] = False
    return m


def dentro_da_imagem(p: np.ndarray, margem: float = 0) -> np.ndarray:
    p = np.atleast_2d(p)
    return ((p[:, 0] > -margem) & (p[:, 0] < LARGURA + margem)
            & (p[:, 1] > -margem) & (p[:, 1] < ALTURA + margem))


def trecho_visivel(Hc: np.ndarray, linha: str) -> np.ndarray:
    a, b = (np.array(p) for p in LINHAS[linha])
    xy = a + np.linspace(0, 1, 200)[:, None] * (b - a)
    return xy[dentro_da_imagem(cv2.perspectiveTransform(xy[None], Hc)[0])]


def ajustar_retas(mascara: np.ndarray, Hc: np.ndarray, faixa: float) -> dict[str, np.ndarray]:
    """Reta (normalizada, em px) de cada linha da quadra, ajustada nos pixels brancos próximos."""
    ys, xs = np.nonzero(mascara)
    pixels = np.column_stack([xs, ys]).astype(np.float64)
    retas = {}
    for linha in LINHAS:
        xy = trecho_visivel(Hc, linha)
        if len(xy) < 10:
            continue
        p0, p1 = cv2.perspectiveTransform(xy[[0, -1]][None], Hc)[0]
        comprimento = np.linalg.norm(p1 - p0)
        u = (p1 - p0) / comprimento
        rel = pixels - p0
        perto = (np.abs(rel @ [-u[1], u[0]]) < faixa) & (rel @ u > -10) & (rel @ u < comprimento + 10)
        if perto.sum() < 40:
            continue
        vx, vy, x0, y0 = cv2.fitLine(pixels[perto].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        r = reta((x0, y0), (x0 + vx, y0 + vy))
        retas[linha] = r / np.linalg.norm(r[:2])
    return retas


def refinar(img: np.ndarray, Hc: np.ndarray):
    """Homografia chão → imagem ajustada nas linhas brancas, e as retas medidas."""
    mascara = mascara_linhas(img)
    for faixa in FAIXAS:
        retas = ajustar_retas(mascara, Hc, faixa)
        pares = [(ponto(n), cruzamento(retas[a], retas[b])) for n, (a, b) in PONTOS.items()
                 if a in retas and b in retas]
        pares = [(q, p) for q, p in pares if dentro_da_imagem(p, 20)[0]]
        if len(pares) < 4:
            return None
        Hc, _ = cv2.findHomography(np.array([q for q, _ in pares]), np.array([p for _, p in pares]))
        if Hc is None:
            return None
    return Hc, retas


def referencia() -> tuple[np.ndarray, tuple]:
    """Homografia do frame de referência (a partir dos pontos marcados à mão) e suas features."""
    ref = ler_json(ANN / "calibracao_referencia.json")
    img = saque(ref["serve_id"]).ler(ref["frame"])
    marcados = ref["pontos"]
    Hc, _ = cv2.findHomography(np.array([ponto(n) for n in marcados]), np.array(list(marcados.values()), float))
    return refinar(img, Hc)[0], features(img)


def linhas_do_saque(s: Saque, Hc_ref: np.ndarray, f_ref) -> dict | None:
    """Pontos 3D ao longo de cada linha visível, cada um com a reta medida na imagem."""
    img = s.ler(s.frame)
    H = homografia(f_ref, features(img), min_inliers=30)
    if H is None:
        return None
    r = refinar(img, H @ Hc_ref)
    if r is None:
        return None
    Hc, retas = r
    amostras = []
    for linha, ab in retas.items():
        xy = trecho_visivel(Hc, linha)
        for X, Y in xy[np.linspace(0, len(xy) - 1, 12).astype(int)]:
            amostras.append([X, Y, 0.0, *ab])
    return {"serve_id": s.serve_id, "Hc": Hc, "amostras": np.array(amostras)}


# ---------------------------------------------------------------- B) câmera

def matriz_P(C: np.ndarray, rvec: np.ndarray, f: float) -> np.ndarray:
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    K = np.array([[f, 0, CX], [0, f, CY], [0, 0, 1]])
    return K @ np.column_stack([R, -R @ C])


def projetar(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    x = np.column_stack([X, np.ones(len(X))]) @ P.T
    return x[:, :2] / x[:, 2:]


def residuos(C, rvec, log_f, amostras) -> np.ndarray:
    """Distância (px) entre cada ponto da quadra projetado e a reta medida na imagem."""
    uv = projetar(matriz_P(C, rvec, np.exp(log_f)), amostras[:, :3])
    return np.sum(amostras[:, 3:5] * uv, axis=1) + amostras[:, 5]


def chute(C: np.ndarray, a: dict) -> np.ndarray:
    """Rotação e log(focal) de um saque com a câmera fixa em C, apontando para o centro da imagem."""
    centro = np.linalg.inv(a["Hc"]) @ [CX, CY, 1]
    z = np.r_[centro[:2] / centro[2], 0] - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1])
    x /= np.linalg.norm(x)
    rvec = cv2.Rodrigues(np.vstack([x, np.cross(z, x), z]))[0].ravel()
    ajuste = lambda p: residuos(C, p[:3], p[3], a["amostras"])
    return least_squares(ajuste, np.r_[rvec, np.log(2000)], loss="soft_l1", f_scale=2.0).x


def ajuste_conjunto(As: list[dict]):
    """Uma posição C para todos; rotação e focal por saque."""
    n = len(As)
    limites = np.r_[0, np.cumsum([len(a["amostras"]) for a in As])]
    x0 = np.concatenate([C_INICIAL, *[chute(C_INICIAL, a) for a in As]])

    def todos_residuos(x):
        p = x[3:].reshape(n, 4)
        return np.concatenate([residuos(x[:3], p[i, :3], p[i, 3], a["amostras"]) for i, a in enumerate(As)])

    dependencias = lil_matrix((limites[-1], 3 + 4 * n), dtype=int)
    for i in range(n):
        dependencias[limites[i]:limites[i + 1], :3] = 1
        dependencias[limites[i]:limites[i + 1], 3 + 4 * i:7 + 4 * i] = 1
    r = least_squares(todos_residuos, x0, jac_sparsity=dependencias, loss="soft_l1", f_scale=2.0,
                      x_scale="jac")
    C = r.x[:3]
    por_saque = r.x[3:].reshape(n, 4)
    rms = [float(np.sqrt(np.mean(r.fun[limites[i]:limites[i + 1]] ** 2))) for i in range(n)]
    return C, por_saque, rms


# ---------------------------------------------------------------- uso e visualização

def P_do_frame(serve_id: str, k: int) -> np.ndarray:
    """P de qualquer frame aberto: P_k = H_k⁻¹ · P_contato."""
    z = np.load(PROC / "homografias" / f"{serve_id}.npz")
    i = np.flatnonzero(z["frames"] == k)
    if len(i) == 0:
        raise KeyError(f"frame {k} de {serve_id} não está no plano aberto")
    return np.linalg.inv(z["H"][i[0]]) @ np.load(DIR_CAL / f"{serve_id}.npz")["P"]


CORES = {"topo_rede": (0, 0, 255), "antena_perto": (255, 0, 255), "antena_longe": (255, 0, 255),
         "linha_central": (0, 200, 255)}


def desenhar(img: np.ndarray, P: np.ndarray, texto: str = "") -> np.ndarray:
    out = img.copy()
    for nome, (a, b) in segmentos_3d().items():
        pts = projetar(P, a + np.linspace(0, 1, 50)[:, None] * (b - a))
        cv2.polylines(out, [np.round(pts).astype(np.int32)], False, CORES.get(nome, (0, 255, 0)), 1, cv2.LINE_AA)
    for cor, espessura in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(out, texto, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, espessura)
    return out


def main() -> None:
    saques = {s.serve_id: s for s in saques_validos()}
    Hc_ref, f_ref = referencia()
    with ProcessPoolExecutor(8) as ex:
        resultados = list(ex.map(partial(linhas_do_saque, Hc_ref=Hc_ref, f_ref=f_ref), saques.values()))
    As = [a for a in resultados if a is not None]
    falhas = [sid for sid, a in zip(saques, resultados) if a is None]

    C, por_saque, rms = ajuste_conjunto(As)

    DIR_CAL.mkdir(parents=True, exist_ok=True)
    DIR_VISUAL.mkdir(parents=True, exist_ok=True)
    resumo = {"camera_C_m": np.round(C, 3).tolist(), "saques": {sid: {"ok": False} for sid in falhas}}
    for a, (*rvec, log_f), e in zip(As, por_saque, rms):
        sid, f = a["serve_id"], float(np.exp(log_f))
        P = matriz_P(C, rvec, f)
        np.savez(DIR_CAL / f"{sid}.npz", P=P, Hc=a["Hc"])
        resumo["saques"][sid] = {"ok": e < RMS_MAX, "rms_px": round(e, 2), "f_px": round(f, 1)}
        s = saques[sid]
        cv2.imwrite(str(DIR_VISUAL / f"{sid}.jpg"), desenhar(s.ler(s.frame), P, f"{sid}  f={f:.0f}px  rms={e:.2f}px"))
    salvar_json(PROC / "calibracao.json", resumo)

    ok = sum(v["ok"] for v in resumo["saques"].values())
    print(f"câmera em C = ({C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f}) m")
    print(f"{ok}/{len(saques)} saques calibrados (rms < {RMS_MAX:.0f}px), rms mediano {np.median(rms):.2f}px")


if __name__ == "__main__":
    main()
