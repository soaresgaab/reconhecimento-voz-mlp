# Metadados e Procedência dos Dados - MRNA (Etapa 01)

## 1. Identificação da Base de Dados

- **Nome da Base**: Multilingual Spoken Words Corpus (MSWC)
- **Versão**: 1.0 (Lançamento Oficial / Cohere For AI / Harvard)
- **Idioma**: Português (`pt`)
- **URL Oficial**: https://mlcommons.org/datasets/spoken-words/
- **Repositório Hugging Face / Download**: https://huggingface.co/datasets/parler-tts/mswc
- **Publicação de Referência**: 
  - Mazumder, M., Chitlangia, S., Banbury, C., et al. (2021). *Multilingual Spoken Words Corpus*. Proceedings of the Neural Information Processing Systems (NeurIPS 2021) Track on Datasets and Benchmarks.
- **Licença**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
  - Derivado originalmente do Mozilla Common Voice Corpus.

---

## 2. Locutores e Distribuição

### Locutor 1
- **Origem**: Base Própria (gravações de referência dos Trabalhos Anteriores - MLP e RBF).
- **Gênero**: Masculino (`MALE`).
- **Palavra DIREITA**: 68 amostras (`dir_01.wav` a `dir_68.wav`).
- **Palavra ESQUERDA**: 73 amostras (`esq_01.wav` a `esq_73.wav`).
- **Total**: 141 gravações WAV (16 kHz, Mono, PCM 16-bit).
- **Segmentação silábica prévia**: 6 diretórios silábicos preservados integralmente (`DA`, `DI`, `ES`, `QUER`, `REI`, `TA`).

### Locutor 2
- **Identificador SPEAKER original (MSWC)**: `939991dfcf12d8250739f3b4e5ed47e0dc4bf8ffb69810f4ab853aa12872a811941be35b7e91af5df8350d75d3c61d1301aba694c41deb3f38f1414703bc0461`
- **Gênero**: Masculino (`MALE`).
- **Palavra DIREITA**: 8 amostras (`loc02_dir_001.wav` a `loc02_dir_008.wav`).
- **Palavra ESQUERDA**: 6 amostras (`loc02_esq_001.wav` a `loc02_esq_006.wav`).
- **Total**: 14 gravações (arquivos OPUS originais preservados e convertidos para WAV 16 kHz mono 16-bit).

### Locutor 3
- **Identificador SPEAKER original (MSWC)**: `9d91497cb25399ad6cf6a4d88548fa84ee3c3687363015eb7b12d269cf363dec73496b009c86266b8b7dda93ff257139bab1ddf599719f46ce5cee865a6d80c1`
- **Gênero**: Masculino (`MALE`).
- **Palavra DIREITA**: 7 amostras (`loc03_dir_001.wav` a `loc03_dir_007.wav`).
- **Palavra ESQUERDA**: 6 amostras (`loc03_esq_001.wav` a `loc03_esq_006.wav`).
- **Total**: 13 gravações (arquivos OPUS originais preservados e convertidos para WAV 16 kHz mono 16-bit).

---

## 3. Especificações Técnicas de Conversão

Os 27 arquivos OPUS originais foram convertidos utilizando `FFmpeg` com a parametrização estrita:
- **Taxa de Amostragem (`-ar`)**: 16000 Hz
- **Canais (`-ac`)**: 1 (Mono)
- **Codec de Áudio (`-c:a`)**: `pcm_s16le` (PCM signed 16-bit little-endian)
- **Filtragem / Transformação**: Nenhuma normalização de pico, ruído ou dinâmica aplicada, mantendo total conformidade com a esteira original.

---

## 4. Validação Acústica e Integridade

Todos os 168 arquivos de áudio (141 do Locutor 1 + 14 do Locutor 2 + 13 do Locutor 3) foram verificados quanto a:
1. **Hashes SHA-256**: Nenhuma duplicidade detectada.
2. **Duração**: Todas as amostras possuem duração consistente (MSWC fixado em 1.000s; Locutor 1 variando entre 0.600s e 1.300s).
3. **Presença de Sinal e Dinâmica**: Todos os arquivos possuem amplitude e RMS condizentes com fala humana ativa.
4. **Saturação**: 0 amostras com corte (clipping).
5. **Status Global**: 100% dos arquivos classificados como `VALIDO`.

