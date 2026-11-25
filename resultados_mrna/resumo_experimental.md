# RESUMO EXPERIMENTAL AUDITADO: MRNA SEM COORDENADOR (ETAPAS 03 E 04)

**Data da Auditoria:** 2026-11-24  
**Objetivo Científico:** Investigar a evolução do aprendizado supervisionado mediante incorporação sucessiva de múltiplas redes neurais artificiais (MRNA) especializadas em correção aditiva dos resíduos anteriores, sem coordenador central.

---

## 1. Configuração do Sistema e Arquitetura Selecionada

- **Entrada:** 16 características espectrais FFT (espectro unilateral normalizado em bandas de 500 Hz).
- **Padronização:** Z-score calculada estritamente com as 375 amostras de treinamento.
- **Base Experimental:** 3 locutores, 168 palavras (18 no teste), 504 sílabas (375 treinamento, 75 validação, 54 teste).
- **Classes Silábicas:** DI, REI, TA, ES, QUER, DA (6 classes).
- **Composição das Redes e Auditoria Rigorosa de Parâmetros:**
  - **Rede 1 (MLP):** Topologia [16, 16, 8, 6], ativação ReLU, Softmax, Entropia Cruzada com Adam manual ($lr=0.01$, $\lambda=0.001$).
    - $W_1 \in \mathbb{R}^{16 \times 16} (256) + b_1 \in \mathbb{R}^{1 \times 16} (16) = 272$
    - $W_2 \in \mathbb{R}^{16 \times 8} (128) + b_2 \in \mathbb{R}^{1 \times 8} (8) = 136$
    - $W_3 \in \mathbb{R}^{8 \times 6} (48) + b_3 \in \mathbb{R}^{1 \times 6} (6) = 54$
    - **Total MLP:** **462 parâmetros** (100% treináveis por retropropagação/Adam).
  - **Rede 2 (RBF Residual):** $K=32$ centros via K-Means manual, espalhamento heurístico adaptativo ($\sigma = 0.5$), camada de saída linear ajustada por Ridge Regression ($\lambda = 0.01$, $\alpha_2 = 1.0$).
    - Centros: $32 \times 16 = 512$ (não supervisionados)
    - Sigmas: $32$ (heurísticos adaptativos)
    - Pesos de saída: $W \in \mathbb{R}^{32 \times 6} (192) + b \in \mathbb{R}^{1 \times 6} (6) = 198$ (estimados analiticamente via Ridge)
    - **Total RBF:** **198 pesos ajustados** na saída (742 parâmetros representacionais totais).
  - **Rede 3 (ADALINE Residual):** 16 entradas $\rightarrow$ 6 saídas lineares treinadas pela regra LMS/Delta em batelada ($lr=0.005$, $\lambda=0.001$, 150 épocas, $\alpha_3 = 1.0$).
    - $W \in \mathbb{R}^{16 \times 6} (96) + b \in \mathbb{R}^{1 \times 6} (6) = 102$
    - **Total ADALINE:** **102 parâmetros** (100% treináveis por gradiente LMS).
  - **Rede 4 (ELM Residual):** Projeção aleatória de 32 neurônios ocultos com ativação $\tanh$, pesos de saída lineares via mínimos quadrados regularizados ($\lambda=0.01$, $\alpha_4 = 1.0$).
    - Pesos de entrada: $W_{\mathrm{in}} \in \mathbb{R}^{16 \times 32} (512) + b_{\mathrm{in}} \in \mathbb{R}^{1 \times 32} (32) = 544$ (aleatórios congelados)
    - Pesos de saída com bias: $W_{\mathrm{out}} \in \mathbb{R}^{33 \times 6} = 198$ (estimados analiticamente por pseudoinversa Ridge)
    - **Total ELM:** **198 pesos ajustados** na saída (742 parâmetros funcionais totais).
  - **Consolidação dos Parâmetros:**
    - Parâmetros supervisionados/ajustados na saída: $462 + 198 + 102 + 198 = \mathbf{960}$.
    - Parâmetros funcionais totais da arquitetura: $462 + 742 + 102 + 742 = \mathbf{2.048}$.

---

## 2. Evolução Sucessiva do Aprendizado (Validação — Semente 42)

| Estágio | Composição | CE Treino | Acc Treino | CE Validação | Acc Validação | Norma Média Resíduo $\|R_k\|_2$ |
|---|---|---|---|---|---|---|
| **A** | MLP Isolada | 0.0860 | 97.33% | 0.6078 | 92.00% | 0.0866 |
| **B** | MLP + RBF | 0.0842 | 97.60% | 0.6069 | 92.00% | 0.0854 |
| **C** | MLP + RBF + ADALINE | 0.0841 | 97.60% | 0.6062 | 92.00% | 0.0853 |
| **D** | MLP + RBF + ADALINE + ELM | 0.0833 | 97.33% | **0.6054** | 92.00% | **0.0848** |

---

## 3. Estabilidade Estatística em Três Sementes (42, 43 e 44)

| Estágio | CE Média $\pm$ Desvio | Acc Média $\pm$ Desvio (%) | Detalhamento CE (42 / 43 / 44) |
|---|---|---|---|
| **Estágio A (MLP)** | $0.8772 \pm 0.2722$ | $89.33\% \pm 2.67\%$ | 0.6078 / 1.1520 / 0.8717 |
| **Estágio B (MLP+RBF)** | $0.8762 \pm 0.2718$ | $89.33\% \pm 2.67\%$ | 0.6069 / 1.1503 / 0.8715 |
| **Estágio C (MLP+RBF+Adaline)** | $0.8756 \pm 0.2725$ | $89.33\% \pm 2.67\%$ | 0.6062 / 1.1511 / 0.8694 |
| **Estágio D (MRNA Completa)** | $\mathbf{0.8752 \pm 0.2734}$ | $\mathbf{89.33\% \pm 2.67\%}$ | 0.6054 / 1.1521 / 0.8680 |

*Nota Científica:* A incorporação das redes residuais reduziu a média da entropia cruzada em validação de 0.8772 para 0.8752. Dada a dispersão entre sementes decorrente da inicialização estocástica da MLP ($\sigma \approx 0.27$), essa melhoria é descritiva e não atinge significância estatística em amostras pareadas ($p > 0.05$).

---

## 4. Análise de Ablação Comparativa

Comparação entre a remoção direta de redes congeladas e a ablação com retreinamento dos estágios posteriores:

| Variante | Remoção Direta (Congelada) | Retreinamento Posterior | Conclusão |
|---|---|---|---|
| **MRNA Completa** | CE = **0.6054** (Acc = 92.00%) | CE = **0.6054** (Acc = 92.00%) | Configuração Ótima |
| **Sem ELM** | CE = 0.6062 (Acc = 92.00%) | CE = 0.6062 (Acc = 92.00%) | ELM contribui com $\Delta = -0.0008$ |
| **Sem ADALINE** | CE = 0.6061 (Acc = 92.00%) | CE = 0.6061 (Acc = 92.00%) | ADALINE contribui com $\Delta = -0.0007$ |
| **Sem RBF** | CE = 0.6061 (Acc = 92.00%) | CE = 0.6061 (Acc = 92.00%) | RBF contribui com $\Delta = -0.0007$ |
| **MLP Isolada** | CE = 0.6078 (Acc = 92.00%) | CE = 0.6078 (Acc = 92.00%) | Residual total: $\Delta = -0.0024$ |

---

## 5. Resultados Finais no Conjunto de Teste Reservado (54 Sílabas)

### A. Matriz de Confusão Numérica Exata no Teste (MRNA Selecionada)
Linhas = Classes Verdadeiras; Colunas = Classes Preditas.

```
       DI   REI    TA    ES  QUER    DA   Total
DI    [ 5     2     0     0     2     0 ]    9
REI   [ 0     7     1     0     1     0 ]    9
TA    [ 1     0     6     1     0     1 ]    9
ES    [ 0     0     0     9     0     0 ]    9
QUER  [ 1     2     0     0     6     0 ]    9
DA    [ 0     0     0     0     0     9 ]    9
Total [ 7    11     7    10     9    10 ]   54
```
**Soma da Matriz:** 54 sílabas (conferência 100% exata).

### B. Métricas Auditadas Diretamente da Matriz de Confusão

| Sílaba | TP | FP | FN | Precisão (%) | Revocação (%) | F1-Score (%) |
|---|---|---|---|---|---|---|
| **DI** | 5 | 2 | 4 | 71.43% | 55.56% | 62.50% |
| **REI** | 7 | 4 | 2 | 63.64% | 77.78% | 70.00% |
| **TA** | 6 | 1 | 3 | 85.71% | 66.67% | 75.00% |
| **ES** | 9 | 1 | 0 | 90.00% | 100.00% | 94.74% |
| **QUER** | 6 | 3 | 3 | 66.67% | 66.67% | 66.67% |
| **DA** | 9 | 1 | 0 | 90.00% | 100.00% | 94.74% |
| **Macro Média** | — | — | — | **77.91%** | **77.78%** | **77.27%** |

- **Acurácia Global no Teste:** $42 / 54 = \mathbf{77.78\%}$.
- **Macro-F1 (Média não ponderada dos F1s por classe):** $\mathbf{77.27\%}$.
- **Entropia Cruzada no Teste:** MLP Isolada = 2.7212 | RBF Isolada = 1.4790 | MRNA Final = **2.7182**.

### C. Acurácia Discriminada por Locutor (Sílabas)
- **Locutor 1:** 35 / 42 corretas (**83.33%**).
- **Locutor 2:** 3 / 6 corretas (**50.00%**).
- **Locutor 3:** 4 / 6 corretas (**66.67%**).
- **Acurácia Macro por Locutor:** **66.67%**.

### D. Reconhecimento de Palavras Completas no Teste (18 Palavras)
- **Avaliação Retrospectiva (Segmentos isolados):** 8/18 acertos (**44.44%**).
- **Inferência de Palavras Desconhecidas (WAV inteiro via hipóteses acústicas concorrentes):** **15/18 acertos (83.33%)**.
  - Locutor 1: 12/14 (85.71%)
  - Locutor 2: 1/2 (50.00%)
  - Locutor 3: 2/2 (100.00%)
  - Rejeições ($\theta = -20.0$): 3 palavras
  - Trocas indevidas entre comandos: **0 trocas (0%)**.
