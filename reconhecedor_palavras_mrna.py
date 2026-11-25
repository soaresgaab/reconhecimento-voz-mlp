"""
=============================================================================
PROJETO ACADÊMICO: RECONHECIMENTO DE VOZ VIA MÚLTIPLAS REDES NEURAIS (MRNA)
Módulo: reconhecedor_palavras_mrna.py
Descrição: Reconhecedor Independente de Palavras Inteiras via MRNA
=============================================================================

Este script atende estritamente à especificação do enunciado da disciplina:
"Ao final, um programa de reconhecimento das palavras inteiras deve ser
confeccionado. O trabalho deverá ser executado confeccionando um programa em
linguagem MatLab, Python ou R que leia os arquivos dos fonemas, a fim de
realizar o treinamento, além de outro programa para o reconhecimento das
palavras inteiras."

Funcionamento:
1. Carrega os parâmetros aprendidos da MRNA treinada (MLP + RBF + ADALINE + ELM)
   a partir de 'resultados_mrna/mrna_modelo_treinado.npz';
2. Lê um arquivo de áudio WAV (16.000 Hz, mono PCM 16-bit);
3. Formula duas hipóteses concorrentes de decodificação fonético-temporal:
   - Hipótese DIREITA: segmentos acústicos candidatos para [DI] -> [REI] -> [TA]
   - Hipótese ESQUERDA: segmentos acústicos candidatos para [ES] -> [QUER] -> [DA]
4. Extrai o espectro unilateral FFT em 16 bandas de 500 Hz para cada segmento;
5. Calcula a inferência direta na MRNA aditiva residual sem coordenador;
6. Acumula a soma das log-probabilidades silábicas:
   Score(DIREITA) = log P(DI|s1) + log P(REI|s2) + log P(TA|s3)
   Score(ESQUERDA) = log P(ES|s1) + log P(QUER|s2) + log P(DA|s3)
7. Aplica limiar de rejeição calibrado (padrao: -20.0);
8. Emite a decisão: 'DIREITA', 'ESQUERDA' ou 'NAO_RECONHECIDO'.
"""

import sys
import math
import argparse
from pathlib import Path
import numpy as np
from scipy.io import wavfile

from segmentacao_audio_mrna import ler_wav_pcm, detectar_limites_silabas_mrna

# Mapeamento do domínio fonético
CLASSES = ["DI", "REI", "TA", "ES", "QUER", "DA"]
CLASSE_ID = {c: i for i, c in enumerate(CLASSES)}


def extrair_caracteristicas_fft_sinal(sinal, taxa=16000, n_bandas=16):
    """
    Extrai as 16 características espectrais de magnitude FFT diretamente de
    um array NumPy unidimensional contendo o sinal de áudio temporal.
    """
    sinal = sinal.astype(np.float64)
    if np.max(np.abs(sinal)) > 1.0:
        sinal = sinal / 32768.0
    n = len(sinal)
    if n == 0:
        return np.zeros(n_bandas, dtype=np.float64)
    espectro = np.fft.fft(sinal)
    n_meio = n // 2 + 1
    mag = np.abs(espectro[:n_meio]) / float(n)
    if n % 2 == 0:
        mag[1:-1] *= 2.0
    else:
        mag[1:] *= 2.0
    freq = np.linspace(0.0, taxa / 2.0, n_meio)
    largura = (taxa / 2.0) / n_bandas
    carac = np.zeros(n_bandas, dtype=np.float64)
    for b in range(n_bandas):
        lo, hi = b * largura, (b + 1) * largura
        idx = np.where((freq >= lo) & ((freq <= hi) if b == n_bandas - 1 else (freq < hi)))[0]
        carac[b] = np.mean(mag[idx]) if len(idx) else 0.0
    return carac


def segmentar_palavra_por_hipotese(amostras_norm, sr, palavra_hipotese):
    """
    Segmenta o sinal de áudio nas 3 sílabas constituintes assumindo a
    hipótese acústica informada ('DIREITA' ou 'ESQUERDA').
    """
    res = detectar_limites_silabas_mrna(amostras_norm, sr, palavra_hipotese)
    segmentos = []
    for info in res:
        i_ini = int(round(info["inicio"] * sr))
        i_fim = int(round(info["fim"] * sr))
        segmentos.append(np.array(amostras_norm[i_ini:i_fim], dtype=np.float64))
    return segmentos


class MRNAInferencia:
    """
    Executor de inferência do modelo MRNA aditivo (MLP + RBF + ADALINE + ELM)
    utilizando exclusivamente os pesos sinápticos e estatísticas exportadas.
    """
    def __init__(self, caminho_pesos="resultados_mrna/mrna_modelo_treinado.npz"):
        caminho = Path(caminho_pesos)
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo de pesos não encontrado: {caminho}")

        dados = np.load(str(caminho))
        self.media_tr = dados["media_tr"]
        self.desvio_tr = dados["desvio_tr"]

        # 1. MLP Primária
        self.mlp_W1 = dados["mlp_W1"]
        self.mlp_b1 = dados["mlp_b1"]
        self.mlp_W2 = dados["mlp_W2"]
        self.mlp_b2 = dados["mlp_b2"]
        self.mlp_W3 = dados["mlp_W3"]
        self.mlp_b3 = dados["mlp_b3"]

        # 2. RBF Residual
        self.rbf_centros = dados["rbf_centros"]
        self.rbf_sigmas = dados["rbf_sigmas"]
        self.rbf_W = dados["rbf_W"]
        self.rbf_b = dados["rbf_b"]

        # 3. ADALINE Residual
        self.ada_W = dados["ada_W"]
        self.ada_b = dados["ada_b"]

        # 4. ELM Residual
        self.elm_Win = dados["elm_Win"]
        self.elm_bin = dados["elm_bin"]
        self.elm_Wout = dados["elm_Wout"]

    def prever_silabas(self, X):
        """
        Executa a passagem para frente nas 4 redes e soma os termos residuais.
        X: matriz de características brutas (N, 16)
        Retorna: matriz de probabilidades normalizadas softmax (N, 6)
        """
        # Padronização Z-score com estatísticas de treinamento
        X_p = (X - self.media_tr) / self.desvio_tr

        # 1. Logits da MLP Primária (16 -> 16 [tanh] -> 8 [tanh] -> 6 [linear])
        z1 = np.dot(X_p, self.mlp_W1) + self.mlp_b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.mlp_W2) + self.mlp_b2
        a2 = np.tanh(z2)
        logits_mlp = np.dot(a2, self.mlp_W3) + self.mlp_b3

        # 2. Resíduo RBF (K=32 funções de base radial gaussianas)
        dists = np.linalg.norm(X_p[:, np.newaxis, :] - self.rbf_centros[np.newaxis, :, :], axis=2)
        phi_rbf = np.exp(-(dists ** 2) / (2.0 * (self.rbf_sigmas[np.newaxis, :] ** 2) + 1e-12))
        h_rbf = np.dot(phi_rbf, self.rbf_W) + self.rbf_b

        # 3. Resíduo ADALINE (16 entradas -> 6 saídas lineares)
        h_ada = np.dot(X_p, self.ada_W) + self.ada_b

        # 4. Resíduo ELM (32 nós ocultos com pesos aleatórios congelados)
        z_elm = np.dot(X_p, self.elm_Win) + self.elm_bin
        H_elm = np.tanh(z_elm)
        H_ext = np.hstack([H_elm, np.ones((H_elm.shape[0], 1))])
        h_elm = np.dot(H_ext, self.elm_Wout)

        # Composição Aditiva MRNA sem coordenador central (alpha=1.0)
        logits_mrna = logits_mlp + 1.0 * h_rbf + 1.0 * h_ada + 1.0 * h_elm

        # Softmax numericamente estável
        exp_l = np.exp(logits_mrna - np.max(logits_mrna, axis=1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        return probs

    def reconhecer_arquivo(self, caminho_wav, limiar_rejeicao=-20.0):
        """
        Reconhece uma palavra completa contida em um arquivo WAV de áudio contínuo.
        """
        amostras_int, amostras_norm, sr = ler_wav_pcm(caminho_wav)
        assert sr == 16000, f"Taxa de amostragem deve ser 16000 Hz, obtido: {sr}"

        # Avaliação sob Hipótese DIREITA (DI -> REI -> TA)
        segs_dir = segmentar_palavra_por_hipotese(amostras_norm, sr, "DIREITA")
        X_dir = np.array([extrair_caracteristicas_fft_sinal(s, sr) for s in segs_dir])
        probs_dir = self.prever_silabas(X_dir)
        eps = 1e-12
        p_di = probs_dir[0, CLASSE_ID["DI"]]
        p_rei = probs_dir[1, CLASSE_ID["REI"]]
        p_ta = probs_dir[2, CLASSE_ID["TA"]]
        escore_dir = np.log(p_di + eps) + np.log(p_rei + eps) + np.log(p_ta + eps)

        # Avaliação sob Hipótese ESQUERDA (ES -> QUER -> DA)
        segs_esq = segmentar_palavra_por_hipotese(amostras_norm, sr, "ESQUERDA")
        X_esq = np.array([extrair_caracteristicas_fft_sinal(s, sr) for s in segs_esq])
        probs_esq = self.prever_silabas(X_esq)
        p_es = probs_esq[0, CLASSE_ID["ES"]]
        p_quer = probs_esq[1, CLASSE_ID["QUER"]]
        p_da = probs_esq[2, CLASSE_ID["DA"]]
        escore_esq = np.log(p_es + eps) + np.log(p_quer + eps) + np.log(p_da + eps)

        melhor_escore = max(escore_dir, escore_esq)
        if melhor_escore < limiar_rejeicao:
            decisao = "NAO_RECONHECIDO"
        else:
            decisao = "DIREITA" if escore_dir >= escore_esq else "ESQUERDA"

        return {
            "decisao": decisao,
            "escore_direita": float(escore_dir),
            "escore_esquerda": float(escore_esq),
            "melhor_escore": float(melhor_escore),
            "probs_silabas_direita": [float(p_di), float(p_rei), float(p_ta)],
            "probs_silabas_esquerda": [float(p_es), float(p_quer), float(p_da)]
        }


def main():
    parser = argparse.ArgumentParser(description="Reconhecedor Independente de Palavras Inteiras MRNA")
    parser.add_argument("audio_wav", help="Caminho para o arquivo WAV da palavra completa (16000 Hz, mono)")
    parser.add_argument("--pesos", default="resultados_mrna/mrna_modelo_treinado.npz", help="Arquivo NPZ com pesos MRNA")
    parser.add_argument("--limiar", type=float, default=-20.0, help="Limiar de rejeição de log-probabilidade")
    args = parser.parse_args()

    reconhecedor = MRNAInferencia(args.pesos)
    resultado = reconhecedor.reconhecer_arquivo(args.audio_wav, limiar_rejeicao=args.limiar)

    print("\n" + "=" * 60)
    print("RECONHECIMENTO DE PALAVRA COMPLETA VIA MRNA")
    print("=" * 60)
    print(f"Arquivo analisado: {args.audio_wav}")
    print(f"Decisão Final:     >>> {resultado['decisao']} <<<")
    print(f"Escore DIREITA:    {resultado['escore_direita']:.4f}")
    print(f"Escore ESQUERDA:   {resultado['escore_esquerda']:.4f}")
    print(f"Melhor Escore:     {resultado['melhor_escore']:.4f} (Limiar de rejeição: {args.limiar:.1f})")
    print("-" * 60)
    print(f"Probabilidades sob hipótese DIREITA  (DI, REI, TA):   "
          f"[{resultado['probs_silabas_direita'][0]:.4f}, "
          f"{resultado['probs_silabas_direita'][1]:.4f}, "
          f"{resultado['probs_silabas_direita'][2]:.4f}]")
    print(f"Probabilidades sob hipótese ESQUERDA (ES, QUER, DA): "
          f"[{resultado['probs_silabas_esquerda'][0]:.4f}, "
          f"{resultado['probs_silabas_esquerda'][1]:.4f}, "
          f"{resultado['probs_silabas_esquerda'][2]:.4f}]")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

