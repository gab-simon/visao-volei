# Japão × Brasil — VNL 2023 masculino (VNL-STES, frames em 720p)

Anotações da partida `jpn_bra_menvnl23_1080p` do dataset **VNL-STES**
(Nguyen et al., *VNL-STES: A Benchmark Dataset and Model for Spatiotemporal
Event Spotting*, CVPR Workshops 2025 — <https://hoangqnguyen.github.io/stes/>),
acompanhadas de uma versão dos frames em resolução mais alta.

## Arquivos versionados

| arquivo | origem | conteúdo |
|---|---|---|
| `jpn_bra_menvnl23_1080p.json` | `vnl_1.5/all.json` (cópia sem alterações; idêntica a `train.json`) | 189 rallies: `video`, `num_frames`, `fps` (25), `num_events`, `events` = [`frame`, `label`, `xy`] |
| `labels_csv/jpn_bra_menvnl23_1080p/*.csv` | `vnl_1.5/labels_csv/` (cópia sem alterações) | mesmos eventos em formato tabular |
| `class.txt` | `vnl_1.5/class.txt` | classes: background, block, receive, score, serve, set, spike |
| `frames_720p_manifest.json` | gerado | início de cada rally no vídeo completo e verificação dos frames reextraídos |

Atenção nos CSVs: as colunas `<evento>_x`/`<evento>_y` são `xy × 224` **nos dois
eixos** (grade 224×224, não 398×224). Para coordenadas em pixels prefira
`xy` do JSON multiplicado por (largura, altura) da imagem usada.

Anotações suspeitas: 18 rallies têm 2 eventos `serve`; em alguns o segundo
coincide com um `spike` no mesmo frame (ex.: `rally_000000`, frame 89, é um
ataque na rede, conferido visualmente).

A partida está no split **train** do VNL-STES. Rallies numerados de 0 a 251, com lacunas (189 presentes).

## Frames em 720p (não versionados)

O dataset original distribui os frames em 398×224. Os frames desta partida
foram reextraídos da transmissão completa (YouTube, 1280×720, 25 fps,
02:33:05) para `data/frames_720p/jpn_bra_menvnl23_1080p/rally_NNNNNN/NNNNNN.jpg`,
**com a mesma quantidade e numeração de arquivos** de `frames_224p`. Portanto:

- `events[].frame` aponta para o mesmo instante nas duas versões;
- `events[].xy` é normalizado em [0, 1] e vale para qualquer resolução.

O início de cada rally no vídeo completo foi encontrado por correspondência
visual com os frames originais (NCC em 64×36, busca grossa a 2 fps e
alinhamento fino frame a frame a 25 fps). Para cada rally, `frames_720p_manifest.json`
registra `start_time` (segundos, convenção `ffmpeg -ss`), a quantidade de frames
e a NCC entre cada frame novo e o frame original de 224p (`ncc_mean`, `ncc_min`).

Os frames originais de 224p desta partida ficam em
`data/frames_224p/jpn_bra_menvnl23_1080p/` (copiados do `vnl_1.0.zip`).

Os vídeos e as imagens não são versionados neste repositório.

## Conferir a sincronização

`validar_sincronizacao.py` (raiz do repositório, OpenCV + NumPy, executado com `uv run`) compara
cada frame de 720p com o frame de mesmo índice do dataset de 224p:

```bash
uv run validar_sincronizacao.py                        # todos os rallies, 10 frames/rally (~10 s)
uv run validar_sincronizacao.py --todos-frames         # os 38.908 frames
uv run validar_sincronizacao.py --rally rally_000016 --mostrar      # janela lado a lado
uv run validar_sincronizacao.py --rally rally_000016 --salvar cmp/  # grava JPGs lado a lado
uv run validar_sincronizacao.py --imagem print.png     # acha o frame do dataset de um print do vídeo
uv run validar_sincronizacao.py --imagem print.png --rally rally_000016 --frame 11
```

Frames sincronizados têm NCC ≥ 0,99; a mesma vista da quadra em outro rally
chega a ~0,95, por isso o limiar padrão é 0,97. Único frame abaixo: o frame 0
do `rally_000104` (0,926), que cai numa vinheta de transição.
