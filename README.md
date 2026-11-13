# visao-volei

Projeto de Visão Computacional aplicada ao vôlei (UFPR). Usa o dataset
**VNL-STES** (Nguyen et al., CVPR Workshops 2025 —
[artigo](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Nguyen_VNL-STES_A_Benchmark_Dataset_and_Model_for_Spatiotemporal_Event_Spotting_CVPRW_2025_paper.html),
[dataset](https://hoangqnguyen.github.io/stes/),
[código](https://github.com/hoangqnguyen/spot)) com a partida
**Japão × Brasil, VNL 2023 masculino** (`jpn_bra_menvnl23_1080p`).

O VNL-STES distribui os frames em 398×224, pequeno demais para rastrear a bola.
Aqui os frames da partida foram reextraídos da transmissão completa em
**1280×720**, com a mesma numeração do dataset original. Assim as anotações
(`frame`, `xy`) continuam valendo sem nenhuma conversão.

Como o foco é o saque, só ficam os frames ao redor de cada `serve`: de
**1 s antes a 3 s depois** (frame do serve −25 até +75, a 25 fps), limitado ao
tamanho do rally. São 207 saques em 189 rallies, ou seja, 18.160 frames por
resolução, contra 38.908 no dataset inteiro.

## Estrutura

```text
visao-volei/
├── README.md
├── validar_sincronizacao.py          confere se 720p e 224p estão sincronizados
└── data/
    ├── annotations/                  versionado no git
    │   ├── README.md                 detalhes do formato e da reextração
    │   ├── jpn_bra_menvnl23_1080p.json   anotações originais (189 rallies)
    │   ├── labels_csv/jpn_bra_menvnl23_1080p/rally_NNNNNN.csv
    │   ├── class.txt
    │   ├── serves.json               cada saque: rally, frame, janela, timestamp no vídeo
    │   └── frames_720p_manifest.json     início de cada rally no vídeo + verificação
    ├── frames_224p/jpn_bra_menvnl23_1080p/rally_NNNNNN/NNNNNN.jpg   NÃO versionado
    └── frames_720p/jpn_bra_menvnl23_1080p/rally_NNNNNN/NNNNNN.jpg   NÃO versionado
```

As imagens (cerca de 4 GB) não vão para o git. Elas são distribuídas à parte, em
`jpn_bra_vnl23_720p.zip`, que já contém este README, o script e a pasta
`data/` completa. Basta descompactar e rodar.

## Anotações

- 189 rallies, 25 fps, split `train` do VNL-STES.
- Eventos: `serve` (207), `receive` (263), `set` (227), `spike` (202),
  `block` (74) e `score` (30).
- `events[].frame` é o índice do frame dentro do rally (`NNNNNN.jpg`).
- `events[].xy` é normalizado em [0, 1]: multiplique por (1280, 720) para obter pixels no 720p.
- Cuidado com os CSVs: as coordenadas em pixel estão numa grade 224×224. Prefira o JSON.
- O dataset tem erros de rótulo conhecidos. Exemplo: no `rally_000000`, frame 89, um ataque
  na rede também está rotulado como `serve`. Nenhum saque foi descartado: os suspeitos estão
  marcados em `serves.json` (`flags`). São 7 no mesmo frame de outro evento e 15 rallies
  com dois `serve`.

## Validar a sincronização

Requer [uv](https://docs.astral.sh/uv/). As dependências (`opencv-python`, `numpy`)
estão declaradas no próprio script, e o `uv run` cria o ambiente sozinho na primeira execução:

```bash
uv run validar_sincronizacao.py                         # todos os rallies (~10 s)
uv run validar_sincronizacao.py --rally rally_000016 --mostrar
uv run validar_sincronizacao.py --rally rally_000016 --salvar comparacoes/
uv run validar_sincronizacao.py --imagem print.png      # qual frame do dataset é este print?
```

O script compara cada frame 720p com o frame de mesmo índice do dataset 224p.
Os dois são reduzidos para 64×36 em cinza e comparados por correlação cruzada
normalizada (NCC). Frames sincronizados ficam com NCC ≥ 0,99. A mesma vista da
quadra em outro rally chega a ~0,95, então o limiar padrão é 0,97. Com
`--mostrar` ou `--salvar`, ele gera as imagens lado a lado com a posição anotada
dos eventos.

Resultado atual: 189/189 rallies aprovados. O único frame abaixo do limiar é o
frame 0 do `rally_000104` (0,926), que cai no meio de uma vinheta de transição.

## Origem do material

As anotações e os frames de 224p são do VNL-STES. O vídeo completo é a transmissão
pública da partida no YouTube (1280×720, 25 fps, 02:33:05). Os vídeos não são
redistribuídos. Use este material apenas para fins acadêmicos.
