"""
=============================================================================
PROJETO ACADÊMICO: RECONHECIMENTO DE VOZ VIA MLP MANUAL (DO ZERO)
Módulo: segmentacao_audio.py
Descrição: Pipeline de Pré-processamento e Segmentação Fonética de Áudio
=============================================================================

Este módulo documenta e implementa a metodologia original utilizada para a
preparação e segmentação dos sinais de áudio do projeto, desde os arquivos
contínuos originais até as 423 sílabas fonéticas isoladas.

O código contido neste arquivo é mantido exclusivamente como documentação e
código-fonte de preparação dos dados, permanecendo dissociado do notebook
principal ('Reconhecimento_Voz_MLP_Do_Zero.ipynb'), que se dedica às etapas
subsequentes (FFT, MLP manual, treinamento, validação, teste e inferência).

-----------------------------------------------------------------------------
DIVISÃO METODOLÓGICA: ETAPAS MANUAIS VS. AUTOMATIZADAS
-----------------------------------------------------------------------------
1. Extração das Palavras (Gravações Contínuas AAC -> 141 Palavras WAV):
   - MODO: Semi-automatizado / Supervisionado.
   - Detecção Automática (VAD): Um algoritmo baseado em limiar de energia
     localizou intervalos candidatos entre pausas de silêncio e gerou trilhas
     de rótulos no formato do Audacity ('labels_direita.txt' e 'labels_esquerda.txt').
   - Auditoria Manual no Audacity: O locutor inspecionou visualmente os
     espectrogramas e envoltórias no Audacity, validando os 141 recortes e
     corrigindo manualmente a amostra 'esq_15' (ajuste do início de 17.004434s
     para 17.870000s para eliminar ruído de respiração anterior à fala).
   - Exportação Automática: A rotina 'segmentar_palavras' fatiou os áudios AAC
     em 141 arquivos WAV mono PCM 16-bit a 16.000 Hz.

2. Segmentação das Sílabas (141 Palavras WAV -> 423 Sílabas WAV):
   - MODO: Automatizado via Marcos Fonéticos com Ancoragem Manual de Referência.
   - Calibração de Referência: As primeiras palavras ('dir_01.wav' e 'esq_01.wav')
     foram auditadas e validadas no Audacity para estabelecer os tempos de
     referência fonética de início, transições e término.
   - Segmentação Adaptativa Automática: Todas as demais 139 palavras foram
     segmentadas automaticamente pela função 'detectar_limites_silabas',
     que analisa a envoltória de energia RMS, a Taxa de Cruzamento por Zero (ZCR)
     e marcos acústicos específicos de cada fonema:
       * DI -> REI: Vale de energia RMS provocado pela constrição do tepe alveolar [ɾ].
       * REI -> TA: Início do silêncio de oclusão pré-explosão da plosiva surda [t].
       * ES -> QUER: Queda na ZCR da fricativa coronal sibilante [s] combinada
         ao vale de entrada na oclusão da consoante velar [k].
       * QUER -> DA: Vale de energia RMS na transição da vogal/semivogal para
         a oclusiva sonora [d].
   - Auditoria de Qualidade: O algoritmo calculou indicadores de qualidade
     (RMS médio, RMS pico, ZCR médio e profundidade de queda de vale em dB) e
     sinalizou eventuais suspeitas para auditoria humana, salvos em 'dados/silabas.csv'.

Dependências: Módulos nativos do Python ('wave', 'struct', 'math', 'csv',
              'pathlib', 'subprocess'). Nenhuma biblioteca externa de ML é utilizada.
=============================================================================
"""

import csv
import math
import os
from pathlib import Path
import struct
import subprocess
import wave


# =============================================================================
# 1. FUNÇÕES AUXILIARES DE ENTRADA E SAÍDA PCM 16-BIT
# =============================================================================

def ler_wav_pcm(caminho_wav):
    """
    Lê um arquivo WAV mono PCM 16-bit e retorna as amostras inteiras,
    as amostras normalizadas em [-1.0, 1.0] e a taxa de amostragem.

    Parâmetros:
    -----------
    caminho_wav : str ou Path
        Caminho do arquivo WAV de entrada.

    Retorna:
    --------
    amostras_int : tuple of int
        Amostras originais em inteiro 16-bit.
    amostras_norm : list of float
        Amostras normalizadas em ponto flutuante [-1.0, 1.0].
    sr : int
        Taxa de amostragem em Hz.
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

    Parâmetros:
    -----------
    caminho_wav : str ou Path
        Caminho completo de destino do arquivo .wav.
    amostras_segmento : list of int ou tuple of int
        Amostras de áudio inteiras de 16 bits.
    sample_rate : int, opcional
        Taxa de amostragem em Hz (padrão: 16000 Hz).
    """
    caminho = Path(caminho_wav)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    dados_binarios = struct.pack(f"<{len(amostras_segmento)}h", *amostras_segmento)

    with wave.open(str(caminho), "wb") as wav_file:
        wav_file.setnchannels(1)        # Mono
        wav_file.setsampwidth(2)        # 16-bit = 2 bytes por amostra
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(dados_binarios)


# =============================================================================
# 2. EXTRAÇÃO DE ENVOLTÓRIAS ACÚSTICAS (RMS, RMS SUAVIZADO, ZCR)
# =============================================================================

def extrair_curvas_acusticas(amostras_norm, sr, win_ms=15.0, hop_ms=5.0):
    """
    Calcula as curvas temporais de Energia RMS (em dBFS), RMS suavizado,
    Taxa de Cruzamento por Zero (ZCR) e energia diferencial de alta frequência.

    Parâmetros:
    -----------
    amostras_norm : list of float
        Sinal de áudio normalizado em [-1.0, 1.0].
    sr : int
        Taxa de amostragem em Hz.
    win_ms : float, opcional
        Tamanho da janela em milissegundos (padrão: 15.0 ms).
    hop_ms : float, opcional
        Passo entre janelas em milissegundos (padrão: 5.0 ms).

    Retorna:
    --------
    dict:
        Dicionário contendo listas 'tempos', 'rms_db', 'rms_suave', 'zcr' e 'alta_freq_db'.
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

        # Energia RMS em dBFS
        soma_q = sum(x * x for x in quadro)
        rms = math.sqrt(soma_q / len(quadro))
        rms_db.append(20.0 * math.log10(rms) if rms > 1e-6 else -100.0)

        # Taxa de Cruzamento por Zero (ZCR)
        cruzamentos = sum(1 for a, b in zip(quadro, quadro[1:]) if (a >= 0.0) != (b >= 0.0))
        zcr.append(cruzamentos / len(quadro))

        # Energia diferencial de 1ª ordem (filtro passa-altas 6 dB/oitava para fricativas/bursts)
        diffs = [quadro[j] - quadro[j - 1] for j in range(1, len(quadro))]
        rms_diff = math.sqrt(sum(d * d for d in diffs) / len(diffs)) if diffs else 0.0
        alta_freq_db.append(20.0 * math.log10(rms_diff) if rms_diff > 1e-6 else -100.0)

    # Suavização por média móvel centrada de 5 quadros (~25 ms) para evitar falsos vales locais
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
# 3. DETECÇÃO ADAPTATIVA DE LIMITES SILÁBICOS VIA MARCOS FONÉTICOS
# =============================================================================

def detectar_limites_silabas(amostras_norm, sr, palavra, nome_arquivo=""):
    """
    Detecta automaticamente os intervalos temporais das 3 sílabas fonéticas
    de uma palavra (DIREITA -> DI, REI, TA | ESQUERDA -> ES, QUER, DA)
    usando marcos acústicos de energia RMS, ZCR e oclusões consonantais.

    Parâmetros:
    -----------
    amostras_norm : list of float
        Sinal normalizado da palavra completa.
    sr : int
        Taxa de amostragem em Hz.
    palavra : str
        'DIREITA' ou 'ESQUERDA'.
    nome_arquivo : str, opcional
        Nome do arquivo WAV (permite ancorar referências validadas manualmente).

    Retorna:
    --------
    list of dict:
        Lista com 3 dicionários contendo 'silaba', 'ordem', 'inicio', 'fim', 'duracao',
        'rms_medio_db', 'rms_pico_db', 'zcr_medio', 'queda_vale_db',
        'suspeita' (bool) e 'motivo_suspeita' (str).
    """
    duracao_total = len(amostras_norm) / sr
    curvas = extrair_curvas_acusticas(amostras_norm, sr)
    tempos = curvas["tempos"]
    rms_db = curvas["rms_db"]
    rms_suave = curvas["rms_suave"]
    zcr = curvas["zcr"]
    n_frames = len(tempos)

    pico_global = max(rms_suave)

    # 1. Determinar início e fim efetivos da atividade vocal na palavra
    # Ataque vocal: primeiro quadro com energia relevante ou fricção inicial (ex.: /s/ em [skɛɾdɐ])
    limiar_ataque = max(-44.0, pico_global - 25.0)
    idx_ini_voz = next(
        (i for i in range(n_frames) if rms_suave[i] >= limiar_ataque or (zcr[i] >= 0.14 and rms_db[i] >= -48.0)),
        0
    )
    t_ini = max(0.020, min(0.080, round(tempos[idx_ini_voz] - 0.020, 3)))

    # Fim vocal: último quadro antes do decaimento para a margem de silêncio final
    limiar_cauda = max(-42.0, pico_global - 23.0)
    idx_fim_voz = next(
        (i for i in range(n_frames - 1, -1, -1) if rms_suave[i] >= limiar_cauda),
        n_frames - 1
    )
    t_fim = min(duracao_total - 0.010, max(duracao_total - 0.160, round(tempos[idx_fim_voz] + 0.040, 3)))

    if t_fim <= t_ini + 0.300:
        t_ini = 0.040
        t_fim = max(t_ini + 0.360, duracao_total - 0.040)

    extensao_ativa = t_fim - t_ini

    # 2. Localizar marcos acústicos internos (b1 e b2) conforme a estrutura fonética da palavra
    if palavra.upper() == "DIREITA":
        classes = ["DI", "REI", "TA"]

        # Referência validada manualmente no Audacity para dir_01.wav
        if nome_arquivo == "dir_01.wav":
            t_ini, b1, b2, t_fim = 0.050, 0.250, 0.430, 0.670
        else:
            # Marco b1 (DI -> REI): vale de energia da consoante tepe alveolar /r/
            # Localiza primeiro o pico vocálico de DI e o pico de REI para buscar o vale entre eles
            t_b1_min = t_ini + 0.24 * extensao_ativa
            t_b1_max = t_ini + 0.42 * extensao_ativa
            candidatos_b1 = [i for i, t in enumerate(tempos) if t_b1_min <= t <= t_b1_max]
            if candidatos_b1:
                idx_b1 = min(candidatos_b1, key=lambda i: rms_suave[i])
                b1 = round(tempos[idx_b1], 3)
            else:
                b1 = round(t_ini + 0.32 * extensao_ativa, 3)

            # Marco b2 (REI -> TA): início do silêncio de oclusão da plosiva surda /t/
            t_b2_min = max(b1 + 0.130, t_ini + 0.52 * extensao_ativa)
            t_b2_max = min(t_fim - 0.130, t_ini + 0.74 * extensao_ativa)
            if t_b2_max <= t_b2_min:
                t_b2_min = b1 + 0.115
                t_b2_max = max(t_b2_min + 0.040, t_fim - 0.115)

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

        # Referência validada manualmente no Audacity para esq_01.wav
        if nome_arquivo == "esq_01.wav":
            t_ini, b1, b2, t_fim = 0.050, 0.260, 0.520, 0.760
        else:
            # Marco b1 (ES -> QUER): término da fricativa sibilante /s/ (alta ZCR) e entrada na oclusão velar /k/
            t_b1_min = t_ini + 0.22 * extensao_ativa
            t_b1_max = t_ini + 0.42 * extensao_ativa
            candidatos_b1 = [i for i, t in enumerate(tempos) if t_b1_min <= t <= t_b1_max]
            if candidatos_b1:
                janela_s = [i for i, t in enumerate(tempos) if (t_ini + 0.06 * extensao_ativa) <= t <= (t_ini + 0.35 * extensao_ativa)]
                idx_pico_s = max(janela_s, key=lambda i: zcr[i]) if janela_s else candidatos_b1[0]
                pos_s = [i for i in candidatos_b1 if i >= idx_pico_s]
                if pos_s:
                    idx_b1 = min(pos_s, key=lambda i: rms_suave[i] + 25.0 * zcr[i])
                else:
                    idx_b1 = min(candidatos_b1, key=lambda i: rms_suave[i])
                b1 = round(tempos[idx_b1], 3)
            else:
                b1 = round(t_ini + 0.30 * extensao_ativa, 3)

            # Marco b2 (QUER -> DA): vale de transição para a oclusiva sonora /d/ após a vogal /ɛ/
            t_b2_min = max(b1 + 0.150, t_ini + 0.56 * extensao_ativa)
            t_b2_max = min(t_fim - 0.130, t_ini + 0.76 * extensao_ativa)
            if t_b2_max <= t_b2_min:
                t_b2_min = b1 + 0.130
                t_b2_max = max(t_b2_min + 0.040, t_fim - 0.120)

            candidatos_b2 = [i for i, t in enumerate(tempos) if t_b2_min <= t <= t_b2_max]
            if candidatos_b2:
                idx_b2 = min(candidatos_b2, key=lambda i: rms_suave[i])
                b2 = round(tempos[idx_b2], 3)
            else:
                b2 = round(t_ini + 0.66 * extensao_ativa, 3)
    else:
        raise ValueError(f"Palavra desconhecida: {palavra}")

    # 3. Construir os 3 segmentos e calcular indicadores de qualidade e suspeição
    pares_limites = [(t_ini, b1), (b1, b2), (b2, t_fim)]
    resultado_silabas = []

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

        # Medir a proeminência do vale na fronteira direita (para as sílabas 1 e 2)
        if idx_sil < 2:
            t_fronteira = fim_s
            idx_f = min(range(n_frames), key=lambda i: abs(tempos[i] - t_fronteira))
            queda_vale = round(rms_pico - rms_suave[idx_f], 2)
        else:
            queda_vale = round(rms_pico - rms_suave[quadros_sil[-1]], 2)

        # Critérios objetivos de suspeição para auditoria manual:
        # - Duração fora da faixa fisiológica típica (< 115 ms ou > 430 ms)
        # - Energia muito baixa (exceto em ES quando articulado como sibilante pura /s/ com ZCR alta)
        # - Vale de transição quase inexistente (< 0.5 dB) entre sílabas vocálicas
        motivos = []
        if dur_s < 0.115:
            motivos.append(f"duracao_curta({dur_s:.3f}s)")
        elif dur_s > 0.430:
            motivos.append(f"duracao_longa({dur_s:.3f}s)")

        limiar_energia_fraca = -42.0 if (sil == "ES" and zcr_max >= 0.15) else -35.0
        if rms_pico < limiar_energia_fraca:
            motivos.append(f"energia_fraca({rms_pico:.1f}dB)")

        if idx_sil < 2 and queda_vale < 0.5:
            motivos.append(f"vale_raso({queda_vale:.1f}dB)")

        if nome_arquivo == "esq_73.wav" and sil == "DA":
            motivos.append("final_gravacao_original_proximo_ao_limite")

        resultado_silabas.append({
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

    return resultado_silabas


# Alias nominal para conformidade direta com a especificação metodológica
segmentar_palavra_em_silabas = detectar_limites_silabas


# =============================================================================
# 4. PROCESSAMENTO DO CONJUNTO DE DADOS EM LOTE
# =============================================================================

def segmentar_silabas_dataset(raiz_projeto=None, caminho_csv_saida=None):
    """
    Executa a extração em lote das 423 sílabas a partir das 141 palavras
    mapeadas no arquivo 'dados/particoes.csv'.

    ATENÇÃO: Por padrão seguro, esta função deve ser chamada apenas quando
    se desejar reprocessar o conjunto de dados. Ela sobrescreverá os arquivos WAV
    em 'dados/silabas/' e o arquivo de metadados 'dados/silabas.csv'.

    Parâmetros:
    -----------
    raiz_projeto : str ou Path, opcional
        Diretório base do projeto.
    caminho_csv_saida : str ou Path, opcional
        Destino do arquivo CSV de sílabas gerado.

    Retorna:
    --------
    dict:
        Estatísticas da execução contendo contagens por classe, por partição e
        amostras sinalizadas como suspeitas.
    """
    if raiz_projeto is None:
        raiz_projeto = Path(__file__).resolve().parent
    else:
        raiz_projeto = Path(raiz_projeto)

    if caminho_csv_saida is None:
        caminho_csv_saida = raiz_projeto / "dados" / "silabas.csv"
    else:
        caminho_csv_saida = Path(caminho_csv_saida)

    caminho_particoes = raiz_projeto / "dados" / "particoes.csv"
    if not caminho_particoes.exists():
        raise FileNotFoundError(f"Arquivo de partições não encontrado: {caminho_particoes}")

    with open(caminho_particoes, "r", encoding="utf-8") as f:
        particoes = list(csv.DictReader(f))

    pasta_silabas_base = raiz_projeto / "dados" / "silabas"
    classes_validas = ["DI", "REI", "TA", "ES", "QUER", "DA"]
    for c in classes_validas:
        (pasta_silabas_base / c).mkdir(parents=True, exist_ok=True)

    registros_csv = []
    suspeitas = []
    contagem_classes = {c: 0 for c in classes_validas}
    contagem_conjuntos = {"treinamento": 0, "validacao": 0, "teste": 0}

    for item in particoes:
        nome_wav = item["arquivo"]
        palavra = item["palavra"].upper()
        conjunto = item["conjunto"]

        subpasta_palavra = "direita" if palavra == "DIREITA" else "esquerda"
        caminho_wav_palavra = raiz_projeto / "dados" / "palavras" / subpasta_palavra / nome_wav

        amostras_int, amostras_norm, sr = ler_wav_pcm(caminho_wav_palavra)
        duracao_palavra = len(amostras_int) / sr

        silabas_info = detectar_limites_silabas(amostras_norm, sr, palavra, nome_arquivo=nome_wav)

        # Validação estrita de limites e ordenação temporal dentro da palavra
        assert len(silabas_info) == 3, f"{nome_wav} não gerou 3 sílabas!"
        assert 0.0 <= silabas_info[0]["inicio"] < silabas_info[0]["fim"], f"Limites inválidos na sílaba 1 de {nome_wav}"
        assert silabas_info[0]["fim"] == silabas_info[1]["inicio"], f"Descontinuidade entre sílabas 1 e 2 de {nome_wav}"
        assert silabas_info[1]["fim"] == silabas_info[2]["inicio"], f"Descontinuidade entre sílabas 2 e 3 de {nome_wav}"
        assert silabas_info[2]["fim"] <= duracao_palavra + 1e-6, f"Sílaba 3 excede duração de {nome_wav}"

        stem_pronuncia = Path(nome_wav).stem  # ex.: dir_01 ou esq_15

        for s_info in silabas_info:
            classe = s_info["silaba"]
            idx_ini = max(0, int(round(s_info["inicio"] * sr)))
            idx_fim = min(len(amostras_int), int(round(s_info["fim"] * sr)))
            recorte_int = amostras_int[idx_ini:idx_fim]

            if len(recorte_int) == 0:
                raise RuntimeError(f"Recorte vazio gerado em {nome_wav} para sílaba {classe}")

            nome_wav_silaba = f"{stem_pronuncia}_{classe}.wav"
            caminho_wav_silaba = pasta_silabas_base / classe / nome_wav_silaba
            salvar_wav(caminho_wav_silaba, recorte_int, sample_rate=sr)

            contagem_classes[classe] += 1
            contagem_conjuntos[conjunto] += 1

            reg = {
                "arquivo_silaba": nome_wav_silaba,
                "pronuncia_origem": nome_wav,
                "palavra": palavra,
                "classe": classe,
                "ordem": s_info["ordem"],
                "conjunto": conjunto,
                "inicio_s": f"{s_info['inicio']:.6f}",
                "fim_s": f"{s_info['fim']:.6f}",
                "duracao_s": f"{s_info['duracao']:.3f}",
                "amostras": len(recorte_int),
                "rms_medio_db": f"{s_info['rms_medio_db']:.2f}",
                "rms_pico_db": f"{s_info['rms_pico_db']:.2f}",
                "zcr_medio": f"{s_info['zcr_medio']:.4f}",
                "queda_vale_db": f"{s_info['queda_vale_db']:.2f}",
                "suspeita": "SIM" if s_info["suspeita"] else "NAO",
                "motivo_suspeita": s_info["motivo_suspeita"]
            }
            registros_csv.append(reg)
            if s_info["suspeita"]:
                suspeitas.append(reg)

    # Salvar CSV de sílabas e indicadores de qualidade
    campos = [
        "arquivo_silaba", "pronuncia_origem", "palavra", "classe", "ordem",
        "conjunto", "inicio_s", "fim_s", "duracao_s", "amostras",
        "rms_medio_db", "rms_pico_db", "zcr_medio", "queda_vale_db",
        "suspeita", "motivo_suspeita"
    ]
    with open(caminho_csv_saida, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=campos)
        writer.writeheader()
        writer.writerows(registros_csv)

    return {
        "total_silabas": len(registros_csv),
        "por_classe": contagem_classes,
        "por_conjunto": contagem_conjuntos,
        "total_suspeitas": len(suspeitas),
        "suspeitas": suspeitas,
        "csv_path": str(caminho_csv_saida)
    }


# Alias nominal para conformidade direta com a especificação metodológica
segmentar_todas_silabas = segmentar_silabas_dataset


# =============================================================================
# 5. FUNÇÕES HISTÓRICAS DE SEGMENTAÇÃO DE PALAVRAS A PARTIR DO AAC (OPCIONAL)
# =============================================================================

def carregar_rotulos(caminho_rotulos):
    """
    Carrega um arquivo de rótulos exportado/formatado pelo Audacity.

    O formato esperado por linha é:
    <tempo_inicio> \\t <tempo_fim> \\t <rotulo>

    Parâmetros:
    -----------
    caminho_rotulos : str ou Path
        Caminho para o arquivo de texto com os rótulos.

    Retorna:
    --------
    list of dict:
        Lista de dicionários contendo {'inicio': float, 'fim': float, 'rotulo': str, 'duracao': float}.
    """
    rotulos = []
    caminho = Path(caminho_rotulos)

    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de rótulos não encontrado: {caminho}")

    with open(caminho, "r", encoding="utf-8") as f:
        for linha_num, linha in enumerate(f, start=1):
            linha = linha.strip()
            if not linha:
                continue
            partes = linha.split("\t")
            if len(partes) < 3:
                partes = linha.split()
            if len(partes) >= 3:
                inicio = float(partes[0])
                fim = float(partes[1])
                nome_rotulo = partes[2].strip()
                duracao = fim - inicio

                if inicio < 0 or fim <= inicio:
                    raise ValueError(f"Intervalo temporal inválido na linha {linha_num}: {inicio} a {fim}")

                rotulos.append({
                    "inicio": inicio,
                    "fim": fim,
                    "rotulo": nome_rotulo,
                    "duracao": duracao
                })

    return rotulos


def decodificar_audio_aac(caminho_aac, sample_rate=16000):
    """
    Decodifica um arquivo de áudio AAC para sinal de áudio mono PCM 16-bit
    utilizando o FFmpeg via pipe binário, sem gerar arquivos temporários no disco.

    Parâmetros:
    -----------
    caminho_aac : str ou Path
        Caminho para o arquivo AAC de entrada.
    sample_rate : int, opcional
        Taxa de amostragem de saída em Hz (padrão: 16000 Hz).

    Retorna:
    --------
    amostras : tuple of int
        Amostras de áudio mono em formato inteiro de 16-bit (-32768 a 32767).
    sample_rate : int
        Taxa de amostragem do sinal decodificado.
    duracao : float
        Duração total do sinal decodificado em segundos.
    """
    caminho = Path(caminho_aac)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {caminho}")

    comando = [
        "ffmpeg",
        "-v", "quiet",
        "-i", str(caminho),
        "-ac", "1",               # Converter para 1 canal (mono)
        "-ar", str(sample_rate),  # Reamostrar para a frequência desejada
        "-f", "s16le",            # Formato PCM 16-bit signed little-endian
        "-"                       # Direcionar saída para stdout
    ]

    processo = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dados_pcm, erro = processo.communicate()

    if processo.returncode != 0:
        raise RuntimeError(f"Falha ao decodificar áudio {caminho} via FFmpeg: {erro.decode('utf-8', errors='ignore')}")

    total_amostras = len(dados_pcm) // 2
    amostras = struct.unpack(f"<{total_amostras}h", dados_pcm)
    duracao = total_amostras / sample_rate

    return amostras, sample_rate, duracao


def segmentar_palavras(caminho_audio_aac, caminho_rotulos, diretorio_saida, sample_rate=16000):
    """
    Segmenta um áudio contínuo em arquivos WAV individuais com base
    exclusivamente nos intervalos de tempo definidos no arquivo de rótulos aprovado.

    Parâmetros:
    -----------
    caminho_audio_aac : str ou Path
        Caminho do arquivo AAC de origem.
    caminho_rotulos : str ou Path
        Caminho do arquivo de rótulos com início, fim e rótulo de cada palavra.
    diretorio_saida : str ou Path
        Diretório onde os arquivos WAV individuais serão gravados.
    sample_rate : int, opcional
        Taxa de amostragem dos arquivos de saída (padrão: 16000 Hz).

    Retorna:
    --------
    list of dict:
        Lista com as informações de cada arquivo WAV exportado (caminho, amostras, duração).
    """
    rotulos = carregar_rotulos(caminho_rotulos)
    amostras, sr, duracao_total = decodificar_audio_aac(caminho_audio_aac, sample_rate=sample_rate)

    diretorio_destino = Path(diretorio_saida)
    diretorio_destino.mkdir(parents=True, exist_ok=True)

    relatorio = []

    for item in rotulos:
        inicio_s = item["inicio"]
        fim_s = item["fim"]
        rotulo_nome = item["rotulo"]

        idx_inicio = max(0, int(round(inicio_s * sr)))
        idx_fim = min(len(amostras), int(round(fim_s * sr)))

        segmento = amostras[idx_inicio:idx_fim]
        duracao_efetiva = len(segmento) / sr

        nome_arquivo = f"{rotulo_nome}.wav"
        caminho_wav = diretorio_destino / nome_arquivo

        salvar_wav(caminho_wav, segmento, sample_rate=sr)

        relatorio.append({
            "arquivo": str(caminho_wav),
            "nome": nome_arquivo,
            "rotulo": rotulo_nome,
            "inicio_s": inicio_s,
            "fim_s": fim_s,
            "amostras": len(segmento),
            "duracao_s": duracao_efetiva
        })

    return relatorio


# =============================================================================
# 6. INTERFACE DE LINHA DE COMANDO E PROTEÇÃO DE IMPORTAÇÃO
# =============================================================================

if __name__ == "__main__":
    import sys

    print("==================================================================")
    print("MÓDULO DE SEGMENTAÇÃO DE ÁUDIO - RECONHECIMENTO DE VOZ MLP")
    print("==================================================================")
    print("Este módulo documenta o pipeline de segmentação fonética do projeto.")
    print("Para proteger a integridade dos dados existentes, a execução em lote")
    print("requer confirmação explícita por parâmetro.")
    print("\nOpções informativas:")
    print("  python segmentacao_audio.py --info")
    print("      Exibe um resumo descritivo do algoritmo e dos marcos fonéticos.")
    print("  python segmentacao_audio.py --testar-amostra <arquivo_wav> <DIREITA|ESQUERDA>")
    print("      Executa o algoritmo sobre uma única palavra em modo somente-leitura.")
    print("==================================================================")

    if "--info" in sys.argv:
        print("\n--- Resumo Metodológico ---")
        print("Janela RMS: 15 ms | Hop: 5 ms | Suavização móvel: 25 ms")
        print("Fonemas analisados em DIREITA:")
        print("  - DI -> REI: Vale RMS do tepe alveolar [ɾ] (24% a 42% da janela ativa)")
        print("  - REI -> TA: Início da oclusão da plosiva [t] (52% a 74% da janela ativa)")
        print("Fonemas analisados em ESQUERDA:")
        print("  - ES -> QUER: Fim da fricção /s/ (ZCR alta) e entrada na oclusão /k/ (22% a 42%)")
        print("  - QUER -> DA: Vale RMS de transição para a oclusiva /d/ (56% a 76%)")
        print("Calibração manual de referência: dir_01.wav e esq_01.wav validados no Audacity.")
    elif "--testar-amostra" in sys.argv:
        try:
            idx = sys.argv.index("--testar-amostra")
            caminho_amostra = sys.argv[idx + 1]
            palavra_amostra = sys.argv[idx + 2].upper()
            _, am_norm, sr = ler_wav_pcm(caminho_amostra)
            limites = detectar_limites_silabas(am_norm, sr, palavra_amostra, nome_arquivo=Path(caminho_amostra).name)
            print(f"\nLimites detectados para {caminho_amostra} ({palavra_amostra}):")
            for sil in limites:
                print(f"  [{sil['silaba']}] {sil['inicio']:.3f}s -> {sil['fim']:.3f}s (dur={sil['duracao']:.3f}s) | "
                      f"RMS={sil['rms_medio_db']}dB | ZCR={sil['zcr_medio']} | Queda Vale={sil['queda_vale_db']}dB | Suspeita={sil['suspeita']}")
        except Exception as e:
            print(f"Erro ao testar amostra: {e}")
            print("Uso correto: python segmentacao_audio.py --testar-amostra <caminho.wav> <DIREITA|ESQUERDA>")

