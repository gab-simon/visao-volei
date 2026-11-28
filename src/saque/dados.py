"""Caminhos, constantes e saques válidos."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
DATA = RAIZ / "data"
ANN = DATA / "annotations"
PROC = DATA / "processed"
FRAMES_720 = DATA / "frames_720p" / "jpn_bra_menvnl23_1080p"
FRAMES_TOSS = DATA / "frames_toss"
VIDEO = Path(os.environ.get(
    "SAQUE_VIDEO", Path.home() / "Downloads" / "🇯🇵 JPN vs. 🇧🇷 BRA - Full Match Mens VNL 2023.mp4"))

FPS = 25
LARGURA, ALTURA = 1280, 720
ANTES, DEPOIS = 75, 25                       # janela: −3 s a +1 s do contato
PLACAR = (slice(576, 684), slice(320, 960))  # (linhas, colunas) do placar, fixo na tela


@dataclass(frozen=True)
class Saque:
    serve_id: str
    rally: str
    frame: int
    xy: tuple[float, float]
    start_time: float

    @property
    def janela(self) -> range:
        return range(self.frame - ANTES, self.frame + DEPOIS + 1)

    def tempo(self, k: int) -> float:
        return self.start_time + k / FPS

    def caminho_toss(self, k: int) -> Path:
        return FRAMES_TOSS / self.serve_id / f"{k:+06d}.jpg"

    def caminho(self, k: int) -> Path:
        """Frame k do frames_toss (parte 1) ou, sem ele, do frames_720p do Drive (só k ≥ 0)."""
        toss = self.caminho_toss(k)
        return toss if toss.exists() else FRAMES_720 / self.rally / f"{k:06d}.jpg"

    def existe(self, k: int) -> bool:
        return self.caminho(k).exists()

    def ler(self, k: int) -> np.ndarray:
        img = cv2.imread(str(self.caminho(k)))
        if img is None:
            raise FileNotFoundError(f"frame {k} de {self.serve_id} não existe em frames_toss nem em frames_720p")
        return img


def ler_json(caminho: Path):
    return json.loads(caminho.read_text())


def salvar_json(caminho: Path, obj) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def saques_validos() -> list[Saque]:
    """Saques que são o primeiro evento do rally; os demais são erros de rótulo do dataset."""
    rallies = {r["video"].split("/")[1]: r for r in ler_json(ANN / "jpn_bra_menvnl23_1080p.json")}
    inicio = ler_json(ANN / "frames_720p_manifest.json")["rallies"]
    return [Saque(s["serve_id"], s["rally"], s["frame"], tuple(s["xy"]), inicio[s["rally"]]["start_time"])
            for s in ler_json(ANN / "serves.json")["serves"]
            if s["frame"] <= min(e["frame"] for e in rallies[s["rally"]]["events"])]


def saque(serve_id: str) -> Saque:
    return next(s for s in saques_validos() if s.serve_id == serve_id)
