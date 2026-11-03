# Reconhecimento de Sílabas e Palavras com MLP do Zero

Projeto acadêmico desenvolvido para a disciplina de **Redes Neurais Artificiais** do **Programa de Pós-Graduação em Ciência da Computação (PPGCC)** da **Universidade Federal do Maranhão (UFMA)**.

O objetivo principal é compreender como uma rede neural Perceptron Multicamadas (MLP) aprende e como seu erro se comporta sob diferentes hiperparâmetros, implementando todas as operações matemáticas **do zero com NumPy**, sem utilizar bibliotecas prontas de Machine Learning (como Scikit-Learn, PyTorch, TensorFlow ou Keras).

A rede classifica seis sílabas fonéticas (`DI`, `REI`, `TA`, `ES`, `QUER` e `DA`) a partir de 16 coeficientes espectrais FFT extraídos de gravações de áudio a 16 kHz. As palavras completas (**DIREITA** e **ESQUERDA**) são identificadas posteriormente por um verificador temporal determinístico de sequências (`if/elif/else`).

---

## Estrutura do Projeto

```text
├── Reconhecimento_Voz_MLP_Experimental.ipynb  # Notebook principal: exploração experimental e seleção de modelo
├── Reconhecimento_Voz_MLP_Do_Zero.ipynb       # Notebook de referência com a implementação manual
├── segmentacao_audio.py                      # Algoritmo de segmentação de sílabas por curvas de energia e ZCR
├── dados/                                    # Dados acústicos completos do projeto
│   ├── originais/                            # Gravações originais contínuas em formato AAC
│   ├── palavras/                             # 141 áudios WAV das palavras individuais recortadas
│   ├── silabas/                              # 423 recortes silábicos em WAV (16 kHz, mono)
│   │   ├── DI/ (68 amostras)
│   │   ├── REI/ (68 amostras)
│   │   ├── TA/ (68 amostras)
│   │   ├── ES/ (73 amostras)
│   │   ├── QUER/ (73 amostras)
│   │   └── DA/ (73 amostras)
│   ├── silabas.csv                           # Metadados de cada segmento silábico (ordem, amostras, etc.)
│   └── particoes.csv                         # Divisão estrita por gravação: 318 treino, 63 validação, 42 teste
├── enunciado.pdf                             # Especificação formal do trabalho prático
├── REQUISITOS.md                             # Critérios metodológicos e restrições técnicas
├── requirements.txt                          # Dependências mínimas necessárias (numpy, scipy, ipython)
├── .gitignore                                # Exclusões de ambiente virtual, caches e temporários
└── README.md                                 # Guia rápido de documentação e execução
```

---

## Como Executar

### 1. Execução Local

1. Clone o repositório ou navegue até a pasta do projeto:
   ```bash
   git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
   cd SEU_REPOSITORIO
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Inicie o Jupyter e execute o notebook:
   ```bash
   jupyter notebook Reconhecimento_Voz_MLP_Experimental.ipynb
   ```

### 2. Execução no Google Colab

1. Abra o [Google Colab](https://colab.research.google.com) e faça upload de `Reconhecimento_Voz_MLP_Experimental.ipynb` (ou abra diretamente pelo GitHub).
2. Na **primeira célula de código** do notebook, configure a variável `REPO_URL` com a URL pública do seu repositório:
   ```python
   REPO_URL = "https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git"
   ```
3. Execute o notebook sequencialmente (**Ambiente de execução** $\to$ **Executar tudo**).
   - O repositório público será clonado automaticamente via HTTPS caso os dados ainda não estejam presentes no ambiente do Colab.
   - Não é necessário utilizar Google Drive (`drive.mount()`), tokens ou qualquer tipo de autenticação.

