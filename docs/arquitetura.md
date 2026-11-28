# Arquitetura — altura do lançamento no saque

Objetivo: medir, em metros, até onde a bola sobe no lançamento de cada saque da
partida Japão × Brasil (VNL 2023).

```text
vídeo + anotações ─► [1] janelas ─► [2] plano aberto ─► [3] calibração ─┐
                                                                       ├─► [5] altura ─► [6] métricas
                                           [4] bola (px) ──────────────┘
```

| # | parte | saída | status |
|---|---|---|---|
| 0 | Dados | frames 720p sincronizados com o VNL-STES, `serves.json` | ✅ |
| 1 | Janelas | frames de −3 s a +1 s do contato | ✅ `saque-janelas` |
| 2 | Plano aberto | trecho de câmera aberta + homografias frame → contato | ✅ `saque-planos` |
| 3 | Calibração | matriz `P` de cada frame aberto | ✅ `saque-calibracao` |
| 4 | Bola | posição da bola em pixels, frame a frame | ✅ `saque-bola` |
| 5 | Altura | trajetória 3D e altura máxima | ✅ `saque-bola` |
| 6 | Métricas e validação | CSV por saque, gráficos, conferência manual | ⏳ |

## Saques usados

Dos 207 rótulos `serve`, 26 caem no meio de jogadas e são erros do dataset
(ex.: `rally_000068`, frame 263). Usamos os **181** que são o primeiro evento do
rally (`saques_validos()` em `src/saque/dados.py`).

## 1. Janelas

O rally do dataset começa quase em cima do contato, então o lançamento fica de
fora. Voltamos ao vídeo completo e extraímos de −3 s a +1 s para
`data/frames_toss/<serve_id>/<k>.jpg`, com `k` = índice do frame no rally
(negativo antes do frame 0). Cada extração é conferida contra `frames_720p`:
181/181 no frame exato.

## 2. Plano aberto

A partir do contato, andamos frame a frame enquanto (a) o frame se registra no
vizinho por homografia (SIFT + MAGSAC) e (b) alinhado, continua parecido com o
frame do contato (NCC ≥ 0,80, o que barra as vinhetas). Como a câmera só gira e
dá zoom, essas homografias valem para pontos em qualquer altura.

**Achado:** a transmissão mostra o lançamento quase sempre em close e só abre a
câmera perto do golpe. A mediana de plano aberto antes do contato é **0,44 s**
(102 saques com ≥ 0,4 s, 18 com ≥ 1 s; calibrados e com ≥ 0,4 s: 99). Em geral não veremos o topo do
lançamento, e a parte 5 resolve isso com física.

## 3. Calibração

Usamos a quadra como régua (`src/saque/quadra.py`).

- **A, por saque:** partimos de 8 pontos marcados à mão num frame de referência
  (`data/annotations/calibracao_referencia.json`), levados a cada saque por
  homografia. Depois ajustamos as 6 linhas (laterais, fundos, ataques) nos pixels
  brancos, estreitando a faixa de busca.
- **B, todos juntos:** com um frame só, a focal e a distância se confundem. Como a
  câmera é uma só e não sai do lugar, estimamos **uma posição C para todos** e,
  por saque, a rotação e a focal (`scipy.least_squares`). As hipóteses são pixel
  quadrado, centro óptico no meio e lente sem distorção.

Resultado: câmera em **C = (−0,07; −26,05; 8,63) m**, alinhada com a rede, 26 m
do centro e 8,6 m de altura. **176/181** saques ficam com RMS < 3 px (mediana
1,6 px). Nos frames do lançamento, `P_do_frame(serve_id, k) = H_k⁻¹ · P_contato`.

Checagem rápida com pessoas em pé: o 2º árbitro deu 1,70 m e dois japoneses
caminhando deram 1,68 e 1,78 m.

## 4. Bola

Não foi preciso treinar nada. Na câmera aberta a bola tem ~10 px e, contra a torcida,
aparece bem na **diferença tripla**: os frames k−2 e k+2 são alinhados no frame k pelas
homografias da parte 2, e fica o que mudou em relação aos dois. As manchas pequenas e
compactas perto do sacador são candidatas. Sobram dezenas por frame (torcida, jogadores,
erros de alinhamento), então só ficam as que formam, com vizinhos em k−1 e k+1, um
trecho quase reto e uniforme que de fato se desloca.

Os frames vêm de `data/frames_toss/` (parte 1) quando existe e, se não, de
`data/frames_720p/`. Como o rally do dataset começa no corte para a câmera aberta,
os dois dão o mesmo plano aberto.

## 5. Altura

Em queda livre, a bola segue `X(t) = X₀ + V₀·t + ½·g·t²`, com `g = (0, 0, −9,81)` e
`t = 0` no contato. Cada frame dá 2 equações lineares em (`X₀`, `V₀`), então 3 frames
determinam a parábola, **sem precisar saber onde está o sacador**. Um RANSAC sorteia 3
candidatos de frames diferentes, conta quantos frames têm um candidato a menos de 4 px da
parábola projetada, refina as 100 melhores hipóteses por mínimos quadrados no erro de
reprojeção e fica com a plausível que explica mais frames (mínimo 7). Plausível: no
contato, a bola está perto ou atrás da linha de fundo, entre 1,8 e 4,5 m, sem subir
rápido. Esse filtro só vale depois do refino; com 3 pontos a profundidade é instável e
hipóteses boas parecem absurdas.

A altura máxima sai da parábola (`z` em `t = v_z/g`), mesmo quando só vemos a descida.

**Resultado:** 73 dos 132 saques com ≥ 7 frames abertos e câmera calibrada têm parábola
(mediana de 13 frames, RMS 0,54 px); em 34 o topo aparece na imagem, nos outros é
extrapolado. Conferidos um a um nas imagens, os 73 seguem a bola real até a mão do
sacador. A altura máxima separa os dois tipos de saque:

| | saques | altura máxima (mediana) | altura no contato (mediana) |
|---|---|---|---|
| flutuante (< 5 m) | 28 | 3,48 m | 2,88 m |
| viagem (≥ 5 m) | 45 | 6,62 m | 3,27 m |

Suspeito: `rally_000182_serve0` (1,91 m, RMS 2,3 px), em que a parábola está certa na
imagem mas a profundidade parece errada.

## 6. Métricas e validação

Por saque: altura máxima, altura no contato, tempo de voo visível, time.
Validação:

- resíduo do ajuste da parábola, em px;
- conferência manual num subconjunto;
- altura de jogadores com estatura conhecida, para validar a calibração.

## Código

```text
src/saque/
├── dados.py        caminhos, constantes, saques válidos
├── quadra.py       medidas da quadra
├── registro.py     homografia entre frames (SIFT)
├── janelas.py      parte 1
├── planos.py       parte 2
├── calibracao.py   parte 3
└── bola.py         partes 4 e 5
```

As saídas ficam em `data/processed/` (fora do git).

## Riscos

- **Pouco plano aberto antes do contato:** a parábola precisa de alguns frames
  de queda limpa.
- **Bola pequena e borrada:** o rastreio é o gargalo provável.
- **Lente sem distorção** é uma hipótese; o RMS de 1,6 px indica que está ok no
  centro da quadra.
