"""Homografia entre dois frames da câmera aberta.

A câmera só gira e dá zoom, então a homografia vale para qualquer ponto da cena,
não só os do chão.
"""

from __future__ import annotations

import cv2
import numpy as np

from .dados import ALTURA, LARGURA, PLACAR

_sift = cv2.SIFT_create(3000)


def features(img: np.ndarray, escala: float = 1.0):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mascara = np.full((ALTURA, LARGURA), 255, np.uint8)
    mascara[PLACAR] = 0
    if escala != 1.0:
        g = cv2.resize(g, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
        mascara = cv2.resize(mascara, g.shape[::-1], interpolation=cv2.INTER_NEAREST)
    pontos, desc = _sift.detectAndCompute(g, mascara)
    return np.float32([p.pt for p in pontos]) / escala, desc


def homografia(a, b, min_inliers: int = 60) -> np.ndarray | None:
    """Homografia que leva pixels do frame `a` para o frame `b` (features de cada um)."""
    (pa, da), (pb, db) = a, b
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return None
    pares = [m for m, n in (p for p in cv2.BFMatcher().knnMatch(da, db, k=2) if len(p) == 2)
             if m.distance < 0.75 * n.distance]
    if len(pares) < min_inliers:
        return None
    origem = pa[[m.queryIdx for m in pares]]
    destino = pb[[m.trainIdx for m in pares]]
    H, inliers = cv2.findHomography(origem, destino, cv2.USAC_MAGSAC, 3.0)
    if H is None or inliers.sum() < min_inliers:
        return None
    return H / H[2, 2]
