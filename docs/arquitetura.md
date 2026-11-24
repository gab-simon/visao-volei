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
| 4 | Bola | posição da bola em pixels, frame a frame | ⏳ |
| 5 | Altura | trajetória 3D e altura máxima | ⏳ |
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

Na câmera aberta a bola tem ~6–10 px. A semente é o `xy` anotado no contato; a
partir dela rastreamos para trás com um detector pronto (YOLO, `sports ball`)
restrito a uma região em volta da última posição. Se o detector falhar demais,
anotamos algumas centenas de frames e ajustamos um modelo pequeno.

## 5. Altura

Em queda livre, a bola segue `X(t) = X₀ + V₀·t + ½·g·t²`, com `g = (0, 0, −9,8)`.
São 6 incógnitas (`X₀`, `V₀`), e cada frame dá 2 equações (`P_k · X(t_k)` tem de
cair na bola detectada). Com uns 4 frames já sobram equações, e um ajuste de
mínimos quadrados dá a trajetória 3D, **sem precisar saber onde está o
sacador**. A altura máxima sai direto da parábola (`z + v_z²/2g`), mesmo quando
só vemos a descida.

O ajuste começa no último frame antes do contato e volta no tempo enquanto o
resíduo continuar baixo; o golpe e a mão do sacador quebram a queda livre.

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
└── calibracao.py   parte 3
```

As saídas ficam em `data/processed/` (fora do git).

## Riscos

- **Pouco plano aberto antes do contato:** a parábola precisa de alguns frames
  de queda limpa.
- **Bola pequena e borrada:** o rastreio é o gargalo provável.
- **Lente sem distorção** é uma hipótese; o RMS de 1,6 px indica que está ok no
  centro da quadra.
