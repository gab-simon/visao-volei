"""Quadra de vôlei (FIVB) em metros.

Origem no centro, sob a rede. X no comprimento (−9 esquerda, +9 direita na imagem),
Y na largura (−4,5 lateral perto da câmera, +4,5 longe), Z para cima.
As linhas têm 5 cm e fazem parte da quadra, então o centro delas fica 2,5 cm para dentro.
"""

from __future__ import annotations

import numpy as np

X_FUNDO = 9.0 - 0.025
Y_LATERAL = 4.5 - 0.025
X_ATAQUE = 3.0 - 0.025          # a borda de trás da linha de ataque fica a 3 m do eixo central
ALTURA_REDE = 2.43
ALTURA_ANTENA = ALTURA_REDE + 0.80

LINHAS = {
    "lateral_perto": ((-X_FUNDO, -Y_LATERAL), (X_FUNDO, -Y_LATERAL)),
    "lateral_longe": ((-X_FUNDO, Y_LATERAL), (X_FUNDO, Y_LATERAL)),
    "fundo_esq": ((-X_FUNDO, -Y_LATERAL), (-X_FUNDO, Y_LATERAL)),
    "fundo_dir": ((X_FUNDO, -Y_LATERAL), (X_FUNDO, Y_LATERAL)),
    "ataque_esq": ((-X_ATAQUE, -Y_LATERAL), (-X_ATAQUE, Y_LATERAL)),
    "ataque_dir": ((X_ATAQUE, -Y_LATERAL), (X_ATAQUE, Y_LATERAL)),
}

PONTOS = {
    "canto_esq_perto": ("fundo_esq", "lateral_perto"),
    "canto_esq_longe": ("fundo_esq", "lateral_longe"),
    "canto_dir_longe": ("fundo_dir", "lateral_longe"),
    "canto_dir_perto": ("fundo_dir", "lateral_perto"),
    "ataque_esq_longe": ("ataque_esq", "lateral_longe"),
    "ataque_esq_perto": ("ataque_esq", "lateral_perto"),
    "ataque_dir_longe": ("ataque_dir", "lateral_longe"),
    "ataque_dir_perto": ("ataque_dir", "lateral_perto"),
}


def reta(a, b) -> np.ndarray:
    return np.cross([*a, 1.0], [*b, 1.0])


def cruzamento(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    p = np.cross(r1, r2)
    return p[:2] / p[2]


def ponto(nome: str) -> np.ndarray:
    a, b = PONTOS[nome]
    return cruzamento(reta(*LINHAS[a]), reta(*LINHAS[b]))


def segmentos_3d() -> dict[str, np.ndarray]:
    """Linhas do chão, linha central, topo da rede e antenas, para desenhar."""
    seg = {n: np.array([[*a, 0.0], [*b, 0.0]]) for n, (a, b) in LINHAS.items()}
    seg["linha_central"] = np.array([[0, -Y_LATERAL, 0], [0, Y_LATERAL, 0]], float)
    seg["topo_rede"] = np.array([[0, -4.5, ALTURA_REDE], [0, 4.5, ALTURA_REDE]])
    for lado, y in (("perto", -4.5), ("longe", 4.5)):
        seg[f"antena_{lado}"] = np.array([[0, y, ALTURA_REDE - 1.0], [0, y, ALTURA_ANTENA]])
    return seg
