"""
=============================================================================
PROJETO ACADÊMICO: RECONHECIMENTO DE VOZ VIA MÚLTIPLAS REDES NEURAIS (MRNA)
Módulo: segmentacao_audio_mrna.py
Descrição: Pipeline Multilocutor de Segmentação Acústica e Preparação de Dados
=============================================================================

Este módulo implementa a metodologia de segmentação acústica multilocutor
para o terceiro trabalho acadêmico (MRNA), estendendo com rigor metodológico
o pipeline original de 'segmentacao_audio.py' (desenvolvido para MLP e RBF).

-----------------------------------------------------------------------------
METODOLOGIA E ADAPTAÇÕES MULTILOCUTOR
-----------------------------------------------------------------------------
1. Unidades de Modelagem Acústica:
   - As unidades acústicas são SÍLABAS fonéticas (DI, REI, TA, ES, QUER, DA),
     e NÃO fonemas individuais isolados.
   - Cada palavra é segmentada estritamente em suas 3 sílabas constituintes:
       * DIREITA -> DI (sílaba 1), REI (sílaba 2), TA (sílaba 3)
       * ESQUERDA -> ES (sílaba 1), QUER (sílaba 2), DA (sílaba 3)

2. Princípios Acústicos Reaproveitados:
   - Energia RMS temporal em dBFS (janela de 15 ms, hop de 5 ms).
   - Suavização móvel centrada de 5 quadros (~25 ms).
   - Taxa de Cruzamento por Zero (ZCR) para delimitação de consoantes fricativas
     e sibilantes (/s/).
   - Filtro diferencial de alta frequência para realce de transições consonantais.
   - Marcos fonéticos de transição:
       * DI -> REI: Vale de energia RMS provocado pela constrição do tepe alveolar [ɾ].
       * REI -> TA: Início do silêncio de oclusão pré-explosão da plosiva surda [t].
       * ES -> QUER: Queda da ZCR da fricativa coronal sibilante [s] combinada
         ao vale de entrada na oclusão velar [k].
       * QUER -> DA: Vale de energia RMS na transição vocálica para a oclusiva
         sonora [d].

3. Alterações Metodológicas em Relação ao Pipeline Monolocutor:
   - Remoção de Limites Temporais Rígidos: No pipeline original do Locutor 1,
     havia restrições como min(0.080, ...) e max(duracao - 0.160, ...), adequadas
     às gravações nativas do autor, mas inadequadas aos arquivos de 1.0s do MSWC
     (onde o locutor pode iniciar a fala em 0.20s ou 0.28s).
   - Detecção Dinâmica de Atividade Vocal (VAD Relativo):
     O início (t_ini) e término (t_fim) da fala são calculados dinamicamente a
     partir do pico global de energia da amostra e de limiares de ataque e cauda:
       * limiar_ataque = max(-44.0 dB, pico_global - 25.0 dB)
       * limiar_cauda = max(-42.0 dB, pico_global - 23.0 dB)
       * margem de segurança pré-ataque (-20 ms) e pós-cauda (+40 ms).
   - Janelas Proporcionais à Extensão Ativa (extensao_ativa = t_fim - t_ini):
     As buscas pelos marcos b1 e b2 ocorrem em faixas relativas à porção
     efetivamente falada, adaptando-se naturalmente a diferentes cadências
     e timbres dos locutores 2 e 3 sem enviesar as amostras de teste.
   - Calibração estritamente baseada no conjunto de Treinamento:
     Os limites de tolerância e busca foram verificados unicamente sobre as
     amostras da partição de treinamento dos três locutores.

4. Limitação Metodológica para Inferência Futura:
   - A função de segmentação aqui documentada recebe o rótulo verdadeiro da palavra
     ('palavra'). Isso é padrão metodológico na preparação de dados supervisionados
     de treinamento. No entanto, para reconhecimento de palavras desconhecidas em
     tempo de inferência, o rótulo da palavra não pode ser conhecido a priori.
   - Para inferência posterior, serão avaliadas hipóteses concorrentes de segmentação.

Dependências: Módulos nativos Python ('wave', 'struct', 'math', 'csv', 'pathlib').
=============================================================================
"""

import csv
import math
import os
from pathlib import Path
import struct
import wave


# =============================================================================
# 1. ENTRADA E SAÍDA PCM 16-BIT
# =============================================================================

def ler_wav_pcm(caminho_wav):
    """
    Lê um arquivo WAV mono PCM 16-bit e retorna as amostras inteiras,
    as amostras normalizadas em [-1.0, 1.0] e a taxa de amostragem.
    """
    with wave.open(str(caminho_wav), "rb") as w:
        n_frames = w.getnframes()
        sr = w.getframerate()
        dados_raw = w.readframes(n_frames)
        amostras_int = struct.unpack(f"<{n_frames}h", dados_raw)
        amostras_norm = [s / 32768.0 for s in amostras_int]
    return amostras_int, amostras_norm, sr


def salvar_wav(caminho_wav, amostras_segmento, sample_rate=16000):
    """
    Salva uma sequência de amostras PCM 16-bit em um arquivo WAV padrão.
    """
    caminho = Path(caminho_wav)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    dados_binarios = struct.pack(f"<{len(amostras_segmento)}h", *amostras_segmento)
    with wave.open(str(caminho), "wb") as wav_file:
        wav_file.setnchannels(1)        # Mono
        wav_file.setsampwidth(2)        # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(dados_binarios)


# =============================================================================
# 2. EXTRAÇÃO DE ENVOLTÓRIAS ACÚSTICAS (RMS, RMS SUAVIZADO, ZCR)
# =============================================================================

def extrair_curvas_acusticas(amostras_norm, sr, win_ms=15.0, hop_ms=5.0):
    """
    Calcula curvas temporais de RMS (dBFS), RMS suavizado, ZCR e energia de alta frequência.
    """
    win_len = int(round(win_ms * sr / 1000.0))
    hop_len = int(round(hop_ms * sr / 1000.0))
    n_frames = max(1, (len(amostras_norm) - win_len) // hop_len + 1)

    tempos = []
    rms_db = []
    zcr = []
    alta_freq_db = []

    for i in range(n_frames):
        ini = i * hop_len
        quadro = amostras_norm[ini:ini + win_len]
        t_centro = (ini + win_len / 2.0) / sr
        tempos.append(t_centro)

        # RMS em dBFS
        soma_q = sum(x * x for x in quadro)
        rms = math.sqrt(soma_q / len(quadro))
        rms_db.append(20.0 * math.log10(rms) if rms > 1e-6 else -100.0)

        # Taxa de Cruzamento por Zero (ZCR)
        cruzamentos = sum(1 for a, b in zip(quadro, quadro[1:]) if (a >= 0.0) != (b >= 0.0))
        zcr.append(cruzamentos / len(quadro))

        # Diferencial de 1ª ordem para altas frequências (fricativas/sibilantes)
        diffs = [quadro[j] - quadro[j - 1] for j in range(1, len(quadro))]
        rms_diff = math.sqrt(sum(d * d for d in diffs) / len(diffs)) if diffs else 0.0
        alta_freq_db.append(20.0 * math.log10(rms_diff) if rms_diff > 1e-6 else -100.0)

    # Suavização por média móvel de 5 quadros (~25 ms)
    rms_suave = []
    for i in range(n_frames):
        viz = rms_db[max(0, i - 2):min(n_frames, i + 3)]
        rms_suave.append(sum(viz) / len(viz))

    return {
        "tempos": tempos,
        "rms_db": rms_db,
        "rms_suave": rms_suave,
        "zcr": zcr,
        "alta_freq_db": alta_freq_db
    }


# =============================================================================
# 3. SEGMENTAÇÃO ACÚSTICA ADAPTATIVA MULTILOCUTOR
# =============================================================================

def detectar_limites_silabas_mrna(amostras_norm, sr, palavra, nome_arquivo=""):
    """
    Detecta de forma adaptativa os limites silábicos para múltiplos locutores,
    eliminando travas artificiais e utilizando janelas relativas à atividade de fala.

    Parâmetros:
    -----------
    amostras_norm : list of float
        Sinal de áudio normalizado em [-1.0, 1.0].
    sr : int
        Taxa de amostragem em Hz.
    palavra : str
        'DIREITA' ou 'ESQUERDA'.
    nome_arquivo : str, opcional
        Nome de identificação do arquivo.

    Retorna:
    --------
    list of dict:
        Metadados acústicos e temporais de cada uma das 3 sílabas detectadas.
    """
    duracao_total = len(amostras_norm) / sr
    curvas = extrair_curvas_acusticas(amostras_norm, sr)
    tempos = curvas["tempos"]
    rms_db = curvas["rms_db"]
    rms_suave = curvas["rms_suave"]
    zcr = curvas["zcr"]
    n_frames = len(tempos)

    pico_global = max(rms_suave)

    # 1. Detecção adaptativa do início (t_ini) e fim (t_fim) da fala ativa
    limiar_ataque = max(-44.0, pico_global - 25.0)
    idx_ini_voz = next(
        (i for i in range(n_frames) if rms_suave[i] >= limiar_ataque or (zcr[i] >= 0.14 and rms_db[i] >= -48.0)),
        0
    )
    t_ini = max(0.010, round(tempos[idx_ini_voz] - 0.020, 3))

    limiar_cauda = max(-42.0, pico_global - 23.0)
    idx_fim_voz = next(
        (i for i in range(n_frames - 1, -1, -1) if rms_suave[i] >= limiar_cauda),
        n_frames - 1
    )
    t_fim = min(duracao_total - 0.010, round(tempos[idx_fim_voz] + 0.040, 3))

    # Proteção de extensão mínima
    if t_fim <= t_ini + 0.250:
        t_ini = max(0.010, round(tempos[0], 3))
        t_fim = min(duracao_total - 0.010, round(tempos[-1], 3))

    extensao_ativa = t_fim - t_ini

    # 2. Localização adaptativa dos marcos internos b1 e b2
    if palavra.upper() == "DIREITA":
        classes = ["DI", "REI", "TA"]

        # Marco b1 (DI -> REI): vale RMS do tepe /r/ entre vogais /i/ e /e/
        t_b1_min = t_ini + 0.20 * extensao_ativa
        t_b1_max = t_ini + 0.45 * extensao_ativa
        candidatos_b1 = [i for i, t in enumerate(tempos) if t_b1_min <= t <= t_b1_max]
        if candidatos_b1:
            idx_b1 = min(candidatos_b1, key=lambda i: rms_suave[i])
            b1 = round(tempos[idx_b1], 3)
        else:
            b1 = round(t_ini + 0.32 * extensao_ativa, 3)

        # Marco b2 (REI -> TA): início da oclusão da consoante plosiva surda /t/
        t_b2_min = max(b1 + 0.080, t_ini + 0.50 * extensao_ativa)
        t_b2_max = min(t_fim - 0.080, t_ini + 0.76 * extensao_ativa)
        if t_b2_max <= t_b2_min:
            t_b2_min = b1 + 0.070
            t_b2_max = max(t_b2_min + 0.030, t_fim - 0.070)

        candidatos_b2 = [i for i, t in enumerate(tempos) if t_b2_min <= t <= t_b2_max]
        if candidatos_b2:
            idx_min_oclusao = min(candidatos_b2, key=lambda i: rms_suave[i])
            nivel_vale = rms_suave[idx_min_oclusao]
            idx_entrada = next(
                (i for i in candidatos_b2 if i <= idx_min_oclusao and rms_suave[i] <= nivel_vale + 3.5),
                idx_min_oclusao
            )
            b2 = round(tempos[idx_entrada], 3)
        else:
            b2 = round(t_ini + 0.62 * extensao_ativa, 3)

    elif palavra.upper() == "ESQUERDA":
        classes = ["ES", "QUER", "DA"]

        # Marco b1 (ES -> QUER): transição sibilante /s/ para oclusão velar /k/
        t_b1_min = t_ini + 0.20 * extensao_ativa
        t_b1_max = t_ini + 0.45 * extensao_ativa
        candidatos_b1 = [i for i, t in enumerate(tempos) if t_b1_min <= t <= t_b1_max]
        if candidatos_b1:
            janela_s = [i for i, t in enumerate(tempos) if (t_ini + 0.05 * extensao_ativa) <= t <= (t_ini + 0.35 * extensao_ativa)]
            idx_pico_s = max(janela_s, key=lambda i: zcr[i]) if janela_s else candidatos_b1[0]
            pos_s = [i for i in candidatos_b1 if i >= idx_pico_s]
            if pos_s:
                idx_b1 = min(pos_s, key=lambda i: rms_suave[i] + 25.0 * zcr[i])
            else:
                idx_b1 = min(candidatos_b1, key=lambda i: rms_suave[i])
            b1 = round(tempos[idx_b1], 3)
        else:
            b1 = round(t_ini + 0.30 * extensao_ativa, 3)

        # Marco b2 (QUER -> DA): transição para a oclusiva /d/ após o núcleo vocálico
        t_b2_min = max(b1 + 0.090, t_ini + 0.52 * extensao_ativa)
        t_b2_max = min(t_fim - 0.080, t_ini + 0.78 * extensao_ativa)
        if t_b2_max <= t_b2_min:
            t_b2_min = b1 + 0.070
            t_b2_max = max(t_b2_min + 0.030, t_fim - 0.070)

        candidatos_b2 = [i for i, t in enumerate(tempos) if t_b2_min <= t <= t_b2_max]
        if candidatos_b2:
            idx_b2 = min(candidatos_b2, key=lambda i: rms_suave[i])
            b2 = round(tempos[idx_b2], 3)
        else:
            b2 = round(t_ini + 0.66 * extensao_ativa, 3)
    else:
        raise ValueError(f"Palavra não suportada: {palavra}")

    # 3. Formatação e auditoria de qualidade
    pares_limites = [(t_ini, b1), (b1, b2), (b2, t_fim)]
    resultado = []

    for idx_sil, (sil, (ini_s, fim_s)) in enumerate(zip(classes, pares_limites)):
        dur_s = round(fim_s - ini_s, 3)
        quadros_sil = [i for i, t in enumerate(tempos) if ini_s <= t <= fim_s]
        if not quadros_sil:
            quadros_sil = [min(range(n_frames), key=lambda i: abs(tempos[i] - (ini_s + fim_s) / 2.0))]

        rms_vals = [rms_db[i] for i in quadros_sil]
        zcr_vals = [zcr[i] for i in quadros_sil]
        rms_medio = sum(rms_vals) / len(rms_vals)
        rms_pico = max(rms_vals)
        zcr_medio = sum(zcr_vals) / len(zcr_vals)
        zcr_max = max(zcr_vals)

        # Proeminência de vale de transição
        if idx_sil < 2:
            t_fronteira = fim_s
            idx_f = min(range(n_frames), key=lambda i: abs(tempos[i] - t_fronteira))
            queda_vale = round(rms_pico - rms_suave[idx_f], 2)
        else:
            queda_vale = round(rms_pico - rms_suave[quadros_sil[-1]], 2)

        # Critérios de auditoria acústica
        motivos = []
        if dur_s < 0.070:
            motivos.append(f"duracao_muito_curta({dur_s:.3f}s)")
        elif dur_s > 0.480:
            motivos.append(f"duracao_muito_longa({dur_s:.3f}s)")

        limiar_energia = -45.0 if (sil == "ES" and zcr_max >= 0.14) else -40.0
        if rms_pico < limiar_energia:
            motivos.append(f"energia_baixa({rms_pico:.1f}dB)")

        if idx_sil < 2 and queda_vale < 0.3:
            motivos.append(f"vale_pouco_definido({queda_vale:.1f}dB)")

        resultado.append({
            "silaba": sil,
            "ordem": idx_sil + 1,
            "inicio": round(ini_s, 6),
            "fim": round(fim_s, 6),
            "duracao": dur_s,
            "rms_medio_db": round(rms_medio, 2),
            "rms_pico_db": round(rms_pico, 2),
            "zcr_medio": round(zcr_medio, 4),
            "queda_vale_db": queda_vale,
            "suspeita": len(motivos) > 0,
            "motivo_suspeita": ";".join(motivos) if motivos else "OK"
        })

    return resultado

