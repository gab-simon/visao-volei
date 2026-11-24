# Relatório — altura do lançamento no saque

Diário curto do trabalho, atualizado a cada parte. A arquitetura completa está
em [`arquitetura.md`](arquitetura.md).

## Parte 0 — De onde vêm os dados

A ideia do trabalho é medir, em metros, a altura do lançamento da bola no
saque. Para isso precisávamos de vídeo de vôlei com os saques já marcados.

Partimos do **VNL-STES** (Nguyen et al., CVPR Workshops 2025), um dataset de
partidas da Liga das Nações com cada evento anotado (saque, recepção,
levantamento, ataque, bloqueio, ponto), indicando o frame e a posição na imagem.
Escolhemos a partida **Japão × Brasil, VNL 2023 masculino**: são 189 rallies e
207 saques anotados.

- Artigo: <https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Nguyen_VNL-STES_A_Benchmark_Dataset_and_Model_for_Spatiotemporal_Event_Spotting_CVPRW_2025_paper.pdf>
- Dataset original: <https://hoangqnguyen.github.io/stes/>
- Vídeo da partida (YouTube): <https://www.youtube.com/watch?v=oyQWrHuSLmc>
- Nossos dados (frames 720p + anotações): <https://drive.google.com/file/d/1J3GHH0t2LxuR-5iOokzTHsZ8G2uNaleb/view?usp=sharing>

O problema foi a qualidade. O dataset distribui os frames em 398×224, e nessa
resolução a bola, quando o sacador está no fundo da quadra, vira um ou dois
pixels. Não dá para medir nada.

Então baixamos a transmissão original da partida no YouTube (1280×720, 25 fps,
2h33) e reencontramos nela cada rally do dataset, comparando as imagens
automaticamente. Depois recortamos os frames de novo, agora em 720p, **com a
mesma numeração do dataset**. Com isso as anotações originais valem sem
nenhuma conversão. O script `validar_sincronizacao.py` confere isso: os 189
rallies batem, com similaridade acima de 0,99 entre o frame novo e o original.

Para não carregar 4 GB à toa, ficamos só com os frames em volta de cada saque
(1 s antes, 3 s depois).

### O que já apareceu olhando os dados

- **O dataset tem erros.** Dos 207 rótulos de saque, 26 estão no meio de uma
  jogada, não no começo do rally. Vamos usar os 181 que fazem sentido.
- **O lançamento costuma ficar fora do recorte.** O rally do dataset começa
  quase em cima do contato, então o toss fica antes do frame 0. Como temos o
  vídeo completo sincronizado, dá para voltar alguns segundos.
- **Nem todo lançamento aparece na câmera aberta.** Às vezes a transmissão
  mostra o sacador de perto e só abre a imagem na hora do golpe. Nesses casos
  não há quadra visível para calibrar a câmera. Vamos ter que separar esses
  saques.

## Parte 1 — Voltando no tempo

Como o lançamento acontece antes do começo dos rallies do dataset, voltamos
ao vídeo completo e recortamos de 3 s antes a 1 s depois de cada golpe. Foram
181 saques e cerca de 18 mil frames. Um detalhe chato: o `ffmpeg` entregava
sempre o frame anterior ao pedido. Só percebemos porque cada extração é
comparada com os frames já validados. Corrigido, os 181 bateram no frame exato.

## Parte 2 — Quando a câmera está aberta

Para medir em metros precisamos ver a quadra. Então, para cada saque,
descobrimos por quanto tempo antes do golpe a imagem é a câmera aberta. Para
isso, conferimos frame a frame se a imagem "encaixa" na anterior. Quando
deixa de encaixar, houve corte de câmera. Também barramos as vinhetas da VNL
que passam por cima da imagem.

O resultado foi o maior achado até agora: **a transmissão quase sempre mostra
o lançamento em close** e só abre a câmera perto do golpe. A mediana é de
menos de meio segundo de câmera aberta antes do contato. Na maioria dos
saques, então, não vamos ver a bola no ponto mais alto. A saída é a física:
sabendo a gravidade, alguns frames da bola **descendo** já bastam para
calcular até onde ela subiu.

## Parte 3 — Calibrando a câmera

Usamos a própria quadra como régua: 18 × 9 m, linhas de ataque a 3 m da rede.
Marcamos à mão 8 pontos num frame, e o resto é automático. O programa
encontra as linhas brancas em cada saque e ajusta a quadra nelas.

A primeira versão calibrava cada saque sozinho e errava feio na distância da
câmera (de 30 a 43 m, conforme o saque). O problema é conhecido: numa foto só,
dar zoom de longe ou ficar perto com menos zoom produz imagens muito
parecidas. A virada foi lembrar que **a câmera é uma só, parada no mesmo lugar
da arquibancada a partida inteira**. Ajustando todos os saques juntos, com a
mesma posição da câmera, ela caiu a 26 m do centro da quadra, 8,6 m de altura
e exatamente alinhada com a rede, como se vê no vídeo. O erro ficou em
1,6 px, e 176 dos 181 saques foram calibrados.

Como teste rápido, medimos pessoas em pé: o 2º árbitro deu 1,70 m e dois
japoneses caminhando, 1,68 e 1,78 m. São números plausíveis, mas a validação
de verdade, com jogadores de altura conhecida, vem na parte 5.

Hoje temos 99 saques com pelo menos 0,4 s de câmera aberta e calibrada antes
do golpe, e 152 com pelo menos 0,2 s.

## Simplificando

Revisando o plano, cortamos a etapa de achar o sacador. Com a gravidade
conhecida, alguns frames da bola já bastam para reconstruir a trajetória em 3D,
sem precisar saber onde a pessoa está. O caminho agora é: rastrear a bola (4),
ajustar a parábola e tirar a altura (5), métricas e validação (6).
