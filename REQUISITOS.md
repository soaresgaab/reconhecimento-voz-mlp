# REQUISITOS — Reconhecimento de Sílabas Fonéticas via MLP

## 1. Objetivo

Desenvolver um sistema de reconhecimento de sílabas fonéticas utilizando uma Rede Neural Artificial do tipo MLP (*Multilayer Perceptron*), capaz de identificar as palavras:

- **DIREITA:** DI → REI → TA
- **ESQUERDA:** ES → QUER → DA

O reconhecimento das palavras deverá ocorrer pela identificação sequencial das sílabas aprendidas pela rede neural.

## 2. Preparação dos áudios

- Utilizar os dois arquivos AAC contendo as repetições de DIREITA e ESQUERDA.
- Preservar os áudios originais.
- Separar cada pronúncia em um arquivo individual.
- Segmentar as pronúncias nas seis classes: DI, REI, TA, ES, QUER e DA.
- Armazenar os segmentos em arquivos WAV organizados por classe.
- Verificar a qualidade e a identificação correta de cada segmento.

## 3. Divisão dos dados

Dividir as amostras nas seguintes proporções:

- **Treinamento:** 75%.
- **Validação:** 15%.
- **Teste:** 10%.

A divisão deverá ser realizada por pronúncia completa, antes da segmentação das sílabas, garantindo que uma mesma pronúncia não apareça em conjuntos diferentes.

A distribuição será estratificada por palavra, com arredondamento documentado quando necessário.

O conjunto de teste será reservado exclusivamente para a avaliação final, sem ser utilizado no ajuste do modelo.

## 4. Processamento dos sinais

- Padronizar os sinais de áudio.
- Aplicar a Transformada Rápida de Fourier (FFT).
- Agrupar as frequências do espectro em bandas.
- Calcular a média das amplitudes de cada grupo de frequências.
- Gerar vetores de características com dimensões fixas para a MLP.
- Definir e documentar os parâmetros do processamento espectral.
- Armazenar os sinais de áudio, as taxas de amostragem, as características extraídas e os rótulos em arquivos `.mat`.

## 5. Treinamento da rede neural

- Implementar uma MLP com treinamento supervisionado.
- Utilizar os vetores de características obtidos com FFT como entradas.
- Treinar a rede para classificar as seis sílabas fonéticas.
- Utilizar os conjuntos de treinamento e validação para ajustar o modelo e seus hiperparâmetros.
- Registrar a arquitetura, os parâmetros e a evolução do treinamento.
- Salvar o modelo treinado e os parâmetros necessários para reproduzir as previsões.

## 6. Reconhecimento das palavras

Desenvolver um programa independente do treinamento que:

- Carregue o modelo MLP treinado.
- Receba um áudio contendo uma palavra inteira.
- Aplique o mesmo processamento utilizado no treinamento.
- Identifique as sílabas em sequência temporal.
- Utilize uma máquina de estados para verificar as sequências:
  - DI → REI → TA = **DIREITA**
  - ES → QUER → DA = **ESQUERDA**
- Reinicie a busca quando identificar uma sequência inválida.
- Retorne DIREITA, ESQUERDA ou NÃO RECONHECIDO.

## 7. Avaliação do sistema

- Avaliar a classificação individual das seis sílabas.
- Avaliar o reconhecimento das palavras completas.
- Utilizar os 10% de teste reservados para a avaliação final.
- Calcular acurácia, precisão, recall e F1-score.
- Gerar matrizes de confusão e gráficos de desempenho.
- Registrar os erros e as limitações observadas.

## 8. Tecnologias e desenvolvimento

- **Linguagem:** Python.
- **Ambiente de desenvolvimento:** Google Antigravity.
- **Rede neural:** MLP.
- **Processamento de sinais:** FFT com médias de amplitudes por grupos de frequências.
- **Armazenamento de dados:** arquivos `.mat`.
- **Versionamento:** Git e GitHub.
- **Execução:** Google Colab.

Todo o código-fonte deverá ser modular, organizado e plenamente comentado.

Não será utilizado MATLAB.

## 9. Reprodutibilidade e integração

- Manter o código-fonte e os dados necessários em um repositório público no GitHub.
- Criar um notebook Jupyter (`.ipynb`) compatível com o Google Colab.
- Configurar o notebook para obter automaticamente os arquivos do repositório GitHub.
- Instalar automaticamente as dependências no ambiente remoto do Colab.
- Permitir a execução do processamento dos áudios, treinamento, validação, testes e reconhecimento.
- Fixar as versões das bibliotecas e as configurações necessárias à reprodução.
- Garantir que o sistema funcione em uma sessão limpa do Colab, sem instalações no computador do usuário.
- Não depender de caminhos locais, arquivos privados ou configurações pessoais.

## 10. Entrega final

A entrega será composta por:

1. **Relatório técnico em PDF**, explicando a metodologia, a preparação dos dados, o processamento com FFT, o treinamento da MLP, o reconhecimento das palavras e os resultados obtidos.
2. **Notebook Jupyter (`.ipynb`)**, executável no Google Colab e integrado ao repositório público do GitHub.

O repositório deverá disponibilizar os códigos-fonte comentados, os dados autorizados para publicação, os arquivos `.mat`, o modelo treinado e as instruções necessárias para reproduzir o experimento.

O notebook deverá permitir a execução completa do projeto diretamente pelo navegador, sem exigir instalações locais.