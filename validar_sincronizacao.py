#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["opencv-python>=4.10", "numpy>=1.26"]
# ///
"""Confere se os frames em 720p estão sincronizados com o dataset VNL-STES (224p).

Para cada frame comparado, as duas imagens são reduzidas para 64x36 em escala de
cinza e comparadas por correlação cruzada normalizada (NCC): 1.0 = mesma imagem. Frames sincronizados ficam acima de
0.99; o mesmo enquadramento da quadra em outro rally chega a ~0.95, por isso o
limiar padrão é 0.97. Um rally é aprovado pela mediana dos frames comparados
(uma vinheta de transição no 1º frame não reprova o rally); frames isolados
abaixo do limiar são listados à parte.

Estrutura esperada (pasta ``data`` ao lado deste script, ou ``--dados``)::

    data/annotations/jpn_bra_menvnl23_1080p.json
    data/annotations/frames_720p_manifest.json
    data/frames_224p/jpn_bra_menvnl23_1080p/rally_NNNNNN/NNNNNN.jpg   (dataset original)
    data/frames_720p/jpn_bra_menvnl23_1080p/rally_NNNNNN/NNNNNN.jpg   (reextraído do vídeo)

Exemplos::

    uv run validar_sincronizacao.py                         # todos os rallies, 10 frames por rally
    uv run validar_sincronizacao.py --todos-frames          # todos os ~39 mil frames
    uv run validar_sincronizacao.py --rally rally_000016 --mostrar
    uv run validar_sincronizacao.py --rally rally_000003 --salvar comparacoes/
    uv run validar_sincronizacao.py --imagem print.png      # qual frame do dataset é esse print?
    uv run validar_sincronizacao.py --imagem print.png --rally rally_000003 --frame 11

Dependências (opencv-python, numpy) declaradas acima; o ``uv run`` instala sozinho.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

MATCH = "jpn_bra_menvnl23_1080p"
DESC_SIZE = (64, 36)
PREVIEW = (640, 360)


# ----------------------------------------------------------------- comparação

def descritor(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(img, DESC_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    x -= x.mean()
    return x / max(float(np.linalg.norm(x)), 1e-6)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    return float(descritor(a) @ descritor(b))


def ler(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img


# ----------------------------------------------------------------- dados

class Dados:
    def __init__(self, raiz: Path):
        self.raiz = raiz
        ann = raiz / "annotations"
        self.rallies = {e["video"].split("/")[1]: e
                        for e in json.loads((ann / f"{MATCH}.json").read_text())}
        manifest = ann / "frames_720p_manifest.json"
        self.manifest = json.loads(manifest.read_text())["rallies"] if manifest.exists() else {}
        self.dir224 = raiz / "frames_224p" / MATCH
        self.dir720 = raiz / "frames_720p" / MATCH
        for d in (self.dir224, self.dir720):
            if not d.is_dir():
                sys.exit(f"pasta não encontrada: {d}")

    def n_frames(self, rally: str) -> int:
        d, n = self.dir224 / rally, 0
        while (d / f"{n:06d}.jpg").exists():
            n += 1
        return n

    def f224(self, rally: str, i: int) -> Path:
        return self.dir224 / rally / f"{i:06d}.jpg"

    def f720(self, rally: str, i: int) -> Path:
        return self.dir720 / rally / f"{i:06d}.jpg"

    def tempo_no_video(self, rally: str, i: int) -> float | None:
        m = self.manifest.get(rally)
        return None if m is None else m["start_time"] + i / self.rallies[rally]["fps"]

    def eventos(self, rally: str, i: int) -> list[dict]:
        return [e for e in self.rallies[rally]["events"] if e["frame"] == i]


def fmt_ts(t: float | None) -> str:
    if t is None:
        return "?"
    ms = int(round(t * 1000))
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


# ----------------------------------------------------------------- visualização

def lado_a_lado(dados: Dados, rally: str, i: int, img224, img720, score: float) -> np.ndarray:
    a = cv2.resize(img224, PREVIEW, interpolation=cv2.INTER_CUBIC)
    b = cv2.resize(img720, PREVIEW, interpolation=cv2.INTER_AREA)
    for e in dados.eventos(rally, i):   # posição anotada (xy normalizado) nas duas imagens
        for img in (a, b):
            c = (int(e["xy"][0] * PREVIEW[0]), int(e["xy"][1] * PREVIEW[1]))
            cv2.circle(img, c, 14, (0, 255, 255), 2)
            cv2.putText(img, e["label"], (c[0] + 16, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(a, "dataset 224p", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(b, f"720p  video {fmt_ts(dados.tempo_no_video(rally, i))}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cor = (0, 200, 0) if score >= 0.9 else (0, 0, 255)
    barra = np.zeros((40, PREVIEW[0] * 2, 3), np.uint8)
    cv2.putText(barra, f"{rally} frame {i:06d}   NCC = {score:.4f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
    return np.vstack([np.hstack([a, b]), barra])


class Saida:
    """Mostra em janela (--mostrar) e/ou grava em disco (--salvar)."""

    def __init__(self, mostrar: bool, pasta: Path | None):
        self.mostrar, self.pasta = mostrar, pasta
        if pasta:
            pasta.mkdir(parents=True, exist_ok=True)

    def __call__(self, nome: str, img: np.ndarray) -> bool:
        if self.pasta:
            cv2.imwrite(str(self.pasta / f"{nome}.jpg"), img)
        if self.mostrar:
            try:
                cv2.imshow("validacao (qualquer tecla = proximo, q = sair)", img)
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    return False
            except cv2.error:
                sys.exit("este OpenCV não tem interface gráfica (pacote headless?); use --salvar PASTA")
        return True


# ----------------------------------------------------------------- modos

def amostrar(n: int, k: int | None, eventos: list[int]) -> list[int]:
    if k is None or k >= n:
        return list(range(n))
    idx = set(np.linspace(0, n - 1, k).round().astype(int).tolist())
    idx.update(f for f in eventos if f < n)
    return sorted(idx)


def validar_rallies(dados: Dados, args) -> int:
    rallies = args.rally or sorted(dados.rallies)
    saida = Saida(args.mostrar, Path(args.salvar) if args.salvar else None)
    reprovados, total, piores, abaixo = [], 0, [], []
    print(f"{'rally':<14}{'início no vídeo':>17}{'frames':>8}{'NCC médio':>11}{'NCC mín':>9}  status")
    for rally in rallies:
        if rally not in dados.rallies:
            print(f"{rally}: não existe nas anotações")
            continue
        n = dados.n_frames(rally)
        ev = [e["frame"] for e in dados.rallies[rally]["events"]]
        frames = amostrar(n, None if args.todos_frames else args.amostras, ev)
        scores = []
        for i in frames:
            p720 = dados.f720(rally, i)
            if not p720.exists():
                scores.append(float("nan"))
                continue
            a, b = ler(dados.f224(rally, i)), ler(p720)
            s = ncc(a, b)
            scores.append(s)
            if (args.mostrar or args.salvar) and (i in ev or len(frames) <= 20 or s < args.limiar):
                if not saida(f"{rally}_{i:06d}", lado_a_lado(dados, rally, i, a, b, s)):
                    args.mostrar = saida.mostrar = False
        arr = np.array(scores)
        faltando = int(np.isnan(arr).sum())
        ok = faltando == 0 and np.nanmedian(arr) >= args.limiar
        abaixo += [f"{rally} #{i} ({s:.3f})" for i, s in zip(frames, scores) if s < args.limiar]
        total += len(frames)
        piores.append((float(np.nanmin(arr)), rally, frames[int(np.nanargmin(arr))]))
        if not ok:
            reprovados.append(rally)
        status = "OK" if ok else ("FALTAM FRAMES" if faltando else "REVISAR")
        print(f"{rally:<14}{fmt_ts(dados.tempo_no_video(rally, 0)):>17}{len(frames):>8}"
              f"{np.nanmean(arr):>11.4f}{np.nanmin(arr):>9.4f}  {status}")
    print(f"\n{len(rallies) - len(reprovados)}/{len(rallies)} rallies OK ({total} frames comparados, limiar {args.limiar})")
    piores.sort()
    print("piores frames:", ", ".join(f"{r} #{i} ({s:.3f})" for s, r, i in piores[:3]))
    if abaixo:
        print(f"frames isolados abaixo do limiar ({len(abaixo)}):", ", ".join(abaixo[:10]))
    if reprovados:
        print("revisar:", " ".join(reprovados))
    return 0 if not reprovados else 1


def validar_imagem(dados: Dados, args) -> int:
    """Compara um print/frame arbitrário com o dataset."""
    img = ler(Path(args.imagem))
    if args.rally:
        rally = args.rally[0]
        if args.frame is None:
            sys.exit("informe --frame junto com --rally")
        ref = ler(dados.f224(rally, args.frame))
        s = ncc(img, ref)
        print(f"{rally} frame {args.frame} (vídeo {fmt_ts(dados.tempo_no_video(rally, args.frame))}): "
              f"NCC = {s:.4f} -> {'SINCRONIZADO' if s >= args.limiar else 'NÃO BATE'}")
        Saida(args.mostrar, Path(args.salvar) if args.salvar else None)(
            f"imagem_vs_{rally}_{args.frame:06d}", lado_a_lado(dados, rally, args.frame, ref, img, s))
        return 0 if s >= args.limiar else 1

    # sem rally/frame: procura o frame mais parecido em todo o dataset 224p
    q = descritor(img)
    nomes, descs = [], []
    rallies = sorted(dados.rallies)
    for k, rally in enumerate(rallies, 1):
        for i in range(dados.n_frames(rally)):
            nomes.append((rally, i))
            descs.append(descritor(cv2.imread(str(dados.f224(rally, i)), cv2.IMREAD_GRAYSCALE)))
        print(f"\rindexando dataset 224p: {k}/{len(rallies)} rallies", end="", flush=True)
    print()
    scores = np.stack(descs) @ q
    top = np.argsort(scores)[::-1]
    print("frames mais parecidos:")
    mostrados: list[tuple[str, int]] = []
    for j in top:
        rally, i = nomes[j]
        if any(r == rally and abs(i - f) < 25 for r, f in mostrados):
            continue   # um resultado por trecho
        mostrados.append((rally, i))
        print(f"  {rally} frame {i:06d}  NCC {scores[j]:.4f}  -> vídeo {fmt_ts(dados.tempo_no_video(rally, i))}")
        if len(mostrados) == 5:
            break
    melhor = float(scores[top[0]])
    print("conclusão:", "o print corresponde a um frame do dataset" if melhor >= args.limiar
          else "nenhum frame do dataset bate (fora de rally, replay, ou print com bordas/interface: recorte só o vídeo)")
    rally, i = nomes[top[0]]
    Saida(args.mostrar, Path(args.salvar) if args.salvar else None)(
        f"imagem_vs_{rally}_{i:06d}", lado_a_lado(dados, rally, i, ler(dados.f224(rally, i)), img, melhor))
    return 0 if melhor >= args.limiar else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dados", default=str(Path(__file__).resolve().parent / "data"), help="pasta data/")
    p.add_argument("--rally", nargs="*", help="ex.: rally_000016 (padrão: todos)")
    p.add_argument("--amostras", type=int, default=10, help="frames por rally (+ frames com evento)")
    p.add_argument("--todos-frames", action="store_true", help="compara todos os frames")
    p.add_argument("--limiar", type=float, default=0.97, help="NCC mínimo para considerar sincronizado")
    p.add_argument("--mostrar", action="store_true", help="abre janela com as comparações")
    p.add_argument("--salvar", metavar="PASTA", help="grava as comparações lado a lado em JPG")
    p.add_argument("--imagem", help="print/frame qualquer para comparar com o dataset")
    p.add_argument("--frame", type=int, help="frame do rally (com --imagem e --rally)")
    args = p.parse_args()
    dados = Dados(Path(args.dados))
    rc = validar_imagem(dados, args) if args.imagem else validar_rallies(dados, args)
    if args.mostrar:
        cv2.destroyAllWindows()
    return rc


if __name__ == "__main__":
    sys.exit(main())
