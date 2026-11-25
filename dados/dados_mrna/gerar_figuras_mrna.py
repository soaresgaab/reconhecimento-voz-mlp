import csv
import math
import os
from pathlib import Path
import subprocess
from segmentacao_audio_mrna import ler_wav_pcm, extrair_curvas_acusticas

FIG_DIR = Path("dados/dados_mrna/figuras")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def export_svg_png(svg_str, base_name):
    svg_path = FIG_DIR / f"{base_name}.svg"
    png_path = FIG_DIR / f"{base_name}.png"
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_str)
    # Renderizar PNG a 300 DPI usando magick
    cmd = ["magick", "-density", "300", "-background", "white", str(svg_path), str(png_path)]
    subprocess.run(cmd, check=True)
    print(f"Gerado: {base_name} (.svg e .png)")

# -----------------------------------------------------------------------------
# 1. FIGURA: EXEMPLOS DE FORMA DE ONDA COM LIMITES DE SEGMENTAÇÃO
# -----------------------------------------------------------------------------
def plot_waveform_examples():
    # Amostras representativas para os 3 locutores (DIREITA e ESQUERDA)
    # 6 painéis verticais ou 3x2
    samples = [
        ("Locutor 1 - DIREITA (dir_01)", "dados/dados_mrna/locutores/locutor_1/palavras/direita/dir_01.wav", "DIREITA", [(0.050, 0.250), (0.250, 0.430), (0.430, 0.670)], ["DI", "REI", "TA"]),
        ("Locutor 1 - ESQUERDA (esq_01)", "dados/dados_mrna/locutores/locutor_1/palavras/esquerda/esq_01.wav", "ESQUERDA", [(0.050, 0.260), (0.260, 0.520), (0.520, 0.760)], ["ES", "QUER", "DA"]),
        ("Locutor 2 - DIREITA (loc02_dir_001)", "dados/dados_mrna/locutores/locutor_2/palavras/direita/loc02_dir_001.wav", "DIREITA", [(0.163, 0.388), (0.388, 0.583), (0.583, 0.743)], ["DI", "REI", "TA"]),
        ("Locutor 2 - ESQUERDA (loc02_esq_001)", "dados/dados_mrna/locutores/locutor_2/palavras/esquerda/loc02_esq_001.wav", "ESQUERDA", [(0.138, 0.312), (0.312, 0.618), (0.618, 0.802)], ["ES", "QUER", "DA"]),
        ("Locutor 3 - DIREITA (loc03_dir_001)", "dados/dados_mrna/locutores/locutor_3/palavras/direita/loc03_dir_001.wav", "DIREITA", [(0.168, 0.347), (0.347, 0.613), (0.613, 0.763)], ["DI", "REI", "TA"]),
        ("Locutor 3 - ESQUERDA (loc03_esq_001)", "dados/dados_mrna/locutores/locutor_3/palavras/esquerda/loc03_esq_001.wav", "ESQUERDA", [(0.228, 0.403), (0.403, 0.657), (0.657, 0.783)], ["ES", "QUER", "DA"]),
    ]
    
    W = 1200
    H = 1400
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    svg.append(f'<text x="{W/2}" y="45" font-family="DejaVu Sans, Arial, sans-serif" font-size="22" font-weight="bold" text-anchor="middle" fill="#1e293b">Segmentação Acústica Multilocutor - Formas de Onda e Limites Silábicos</text>')
    svg.append(f'<text x="{W/2}" y="70" font-family="DejaVu Sans, Arial, sans-serif" font-size="14" fill="#64748b" text-anchor="middle">Comparativo dos três locutores para os comandos vocais DIREITA e ESQUERDA</text>')
    
    panel_h = 175
    panel_w = 1040
    start_x = 100
    
    col_bounds = ["#2563eb", "#d97706", "#059669"]
    col_bg_sil = ["#eff6ff", "#fffbeb", "#ecfdf5"]
    
    for i, (title, path, word, bounds, syls) in enumerate(samples):
        y_top = 100 + i * (panel_h + 35)
        _, am_norm, sr = ler_wav_pcm(path)
        dur = len(am_norm) / sr
        
        # Sub-amostrar sinal para renderizar forma de onda limpa em SVG (~1000 pontos)
        step = max(1, len(am_norm) // 800)
        pts_wave = []
        mid_y = y_top + panel_h / 2.0
        scale_y = (panel_h - 40) / 2.0
        
        # Fundo do painel
        svg.append(f'<rect x="{start_x}" y="{y_top}" width="{panel_w}" height="{panel_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4"/>')
        
        # Fatias coloridas das sílabas
        for (b_ini, b_fim), sname, cbg, cbord in zip(bounds, syls, col_bg_sil, col_bounds):
            x1 = start_x + (b_ini / dur) * panel_w
            x2 = start_x + (b_fim / dur) * panel_w
            w_box = max(0, x2 - x1)
            svg.append(f'<rect x="{x1:.1f}" y="{y_top}" width="{w_box:.1f}" height="{panel_h}" fill="{cbg}" opacity="0.65"/>')
            svg.append(f'<line x1="{x1:.1f}" y1="{y_top}" x2="{x1:.1f}" y2="{y_top+panel_h}" stroke="{cbord}" stroke-width="2" stroke-dasharray="4,2"/>')
            svg.append(f'<line x1="{x2:.1f}" y1="{y_top}" x2="{x2:.1f}" y2="{y_top+panel_h}" stroke="{cbord}" stroke-width="2" stroke-dasharray="4,2"/>')
            # Rótulo da sílaba
            svg.append(f'<text x="{(x1+x2)/2:.1f}" y="{y_top+25}" font-family="DejaVu Sans, Arial, sans-serif" font-size="14" font-weight="bold" fill="{cbord}" text-anchor="middle">[{sname}] ({(b_fim-b_ini):.3f}s)</text>')
            
        # Linha central (amplitude zero)
        svg.append(f'<line x1="{start_x}" y1="{mid_y}" x2="{start_x+panel_w}" y2="{mid_y}" stroke="#94a3b8" stroke-width="0.8"/>')
        
        # Desenhar forma de onda
        d_path = []
        for idx in range(0, len(am_norm), step):
            chunk = am_norm[idx:idx+step]
            t_s = (idx + len(chunk)/2) / sr
            x = start_x + (t_s / dur) * panel_w
            max_v = max(chunk)
            min_v = min(chunk)
            y_max = mid_y - max_v * scale_y
            y_min = mid_y - min_v * scale_y
            d_path.append(f"M {x:.1f} {y_min:.1f} L {x:.1f} {y_max:.1f}")
        svg.append(f'<path d="{" ".join(d_path)}" stroke="#334155" stroke-width="1.2"/>')
        
        # Título do painel e eixos
        svg.append(f'<text x="{start_x + 10}" y="{y_top - 8}" font-family="DejaVu Sans, Arial, sans-serif" font-size="13" font-weight="bold" fill="#0f172a">{title} - Duração: {dur:.3f}s</text>')
        svg.append(f'<text x="{start_x - 10}" y="{mid_y + 4}" font-family="DejaVu Sans, Arial, sans-serif" font-size="11" fill="#64748b" text-anchor="end">0.0</text>')
        svg.append(f'<text x="{start_x - 10}" y="{y_top + 15}" font-family="DejaVu Sans, Arial, sans-serif" font-size="11" fill="#64748b" text-anchor="end">+1.0</text>')
        svg.append(f'<text x="{start_x - 10}" y="{y_top + panel_h - 5}" font-family="DejaVu Sans, Arial, sans-serif" font-size="11" fill="#64748b" text-anchor="end">-1.0</text>')
        
        # Tiques de tempo
        n_ticks = 5
        for tk in range(n_ticks + 1):
            t_val = tk * (dur / n_ticks)
            x_tk = start_x + (t_val / dur) * panel_w
            svg.append(f'<line x1="{x_tk:.1f}" y1="{y_top+panel_h}" x2="{x_tk:.1f}" y2="{y_top+panel_h+5}" stroke="#64748b" stroke-width="1"/>')
            svg.append(f'<text x="{x_tk:.1f}" y="{y_top+panel_h+16}" font-family="DejaVu Sans, Arial, sans-serif" font-size="10" fill="#64748b" text-anchor="middle">{t_val:.2f}s</text>')

    svg.append('</svg>')
    export_svg_png("\n".join(svg), "segmentacao_multilocutor_exemplos")

# -----------------------------------------------------------------------------
# 2. FIGURA: CURVAS RMS E ZCR COM LIMITES
# -----------------------------------------------------------------------------
def plot_rms_zcr_curves():
    # Amostra ilustrativa com análise acústica detalhada (Locutor 2 - DIREITA e ESQUERDA)
    W = 1100
    H = 800
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    svg.append(f'<text x="{W/2}" y="45" font-family="DejaVu Sans, Arial, sans-serif" font-size="22" font-weight="bold" text-anchor="middle" fill="#1e293b">Envoltórias Acústicas de Energia RMS e ZCR</text>')
    svg.append(f'<text x="{W/2}" y="70" font-family="DejaVu Sans, Arial, sans-serif" font-size="14" fill="#64748b" text-anchor="middle">Detecção fonética de vales de energia e transições fricativas em amostras multilocutor</text>')
    
    samples = [
        ("Locutor 2: loc02_dir_001.wav (DIREITA)", "dados/dados_mrna/locutores/locutor_2/palavras/direita/loc02_dir_001.wav", [(0.163, 0.388), (0.388, 0.583), (0.583, 0.743)], ["DI", "REI", "TA"]),
        ("Locutor 2: loc02_esq_001.wav (ESQUERDA)", "dados/dados_mrna/locutores/locutor_2/palavras/esquerda/loc02_esq_001.wav", [(0.138, 0.312), (0.312, 0.618), (0.618, 0.802)], ["ES", "QUER", "DA"])
    ]
    
    panel_w = 900
    panel_h = 280
    start_x = 100
    
    for idx, (title, path, bounds, syls) in enumerate(samples):
        y_top = 110 + idx * (panel_h + 80)
        _, am_norm, sr = ler_wav_pcm(path)
        curvas = extrair_curvas_acusticas(am_norm, sr)
        dur = len(am_norm) / sr
        tempos = curvas["tempos"]
        rms_db = curvas["rms_suave"]
        zcr = curvas["zcr"]
        
        svg.append(f'<rect x="{start_x}" y="{y_top}" width="{panel_w}" height="{panel_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4"/>')
        svg.append(f'<text x="{start_x + 10}" y="{y_top - 10}" font-family="DejaVu Sans, Arial, sans-serif" font-size="14" font-weight="bold" fill="#0f172a">{title}</text>')
        
        # Sílabas destacadas
        for (b_ini, b_fim), sname, col in zip(bounds, syls, ["#2563eb", "#d97706", "#059669"]):
            x1 = start_x + (b_ini / dur) * panel_w
            x2 = start_x + (b_fim / dur) * panel_w
            svg.append(f'<rect x="{x1:.1f}" y="{y_top}" width="{(x2-x1):.1f}" height="{panel_h}" fill="{col}" opacity="0.08"/>')
            svg.append(f'<line x1="{x1:.1f}" y1="{y_top}" x2="{x1:.1f}" y2="{y_top+panel_h}" stroke="{col}" stroke-width="1.8" stroke-dasharray="4,2"/>')
            svg.append(f'<line x1="{x2:.1f}" y1="{y_top}" x2="{x2:.1f}" y2="{y_top+panel_h}" stroke="{col}" stroke-width="1.8" stroke-dasharray="4,2"/>')
            svg.append(f'<text x="{(x1+x2)/2:.1f}" y="{y_top + 25}" font-family="DejaVu Sans, Arial, sans-serif" font-size="13" font-weight="bold" fill="{col}" text-anchor="middle">[{sname}]</text>')
            
        # Plot RMS dB (escala -60 a 0 dB)
        min_db, max_db = -60.0, 0.0
        d_rms = []
        for t, val in zip(tempos, rms_db):
            v_clip = max(min_db, min(max_db, val))
            x = start_x + (t / dur) * panel_w
            y = y_top + panel_h - ((v_clip - min_db) / (max_db - min_db)) * panel_h
            d_rms.append(f"{'M' if not d_rms else 'L'} {x:.1f} {y:.1f}")
        svg.append(f'<path d="{" ".join(d_rms)}" fill="none" stroke="#2563eb" stroke-width="2.2"/>')
        
        # Plot ZCR (escala 0.0 a 0.5)
        d_zcr = []
        for t, val in zip(tempos, zcr):
            v_clip = max(0.0, min(0.5, val))
            x = start_x + (t / dur) * panel_w
            y = y_top + panel_h - (v_clip / 0.5) * panel_h
            d_zcr.append(f"{'M' if not d_zcr else 'L'} {x:.1f} {y:.1f}")
        svg.append(f'<path d="{" ".join(d_zcr)}" fill="none" stroke="#dc2626" stroke-width="1.8" stroke-dasharray="3,2"/>')
        
        # Legenda do painel
        svg.append(f'<text x="{start_x - 10}" y="{y_top + 15}" font-family="DejaVu Sans, Arial, sans-serif" font-size="11" fill="#2563eb" text-anchor="end">0 dB</text>')
        svg.append(f'<text x="{start_x - 10}" y="{y_top + panel_h}" font-family="DejaVu Sans, Arial, sans-serif" font-size="11" fill="#2563eb" text-anchor="end">-60 dB</text>')
        svg.append(f'<text x="{start_x + panel_w + 10}" y="{y_top + 15}" font-family="DejaVu Sans, Arial, sans-serif" font-size="11" fill="#dc2626">ZCR 0.5</text>')
        svg.append(f'<text x="{start_x + panel_w + 10}" y="{y_top + panel_h}" font-family="DejaVu Sans, Arial, sans-serif" font-size="11" fill="#dc2626">ZCR 0.0</text>')
        
        # Eixo de tempo
        for tk in range(6):
            t_val = tk * (dur / 5.0)
            x_tk = start_x + (t_val / dur) * panel_w
            svg.append(f'<line x1="{x_tk:.1f}" y1="{y_top+panel_h}" x2="{x_tk:.1f}" y2="{y_top+panel_h+5}" stroke="#64748b" stroke-width="1"/>')
            svg.append(f'<text x="{x_tk:.1f}" y="{y_top+panel_h+16}" font-family="DejaVu Sans, Arial, sans-serif" font-size="10" fill="#64748b" text-anchor="middle">{t_val:.2f}s</text>')

    # Legenda global
    svg.append(f'<rect x="{start_x + panel_w/2 - 140}" y="{H - 45}" width="280" height="30" fill="#ffffff" stroke="#cbd5e1" rx="4"/>')
    svg.append(f'<line x1="{start_x + panel_w/2 - 120}" y1="{H - 30}" x2="{start_x + panel_w/2 - 90}" y2="{H - 30}" stroke="#2563eb" stroke-width="2.5"/>')
    svg.append(f'<text x="{start_x + panel_w/2 - 80}" y="{H - 26}" font-family="DejaVu Sans, Arial, sans-serif" font-size="12" fill="#1e293b">Energia RMS (dBFS)</text>')
    svg.append(f'<line x1="{start_x + panel_w/2 + 50}" y1="{H - 30}" x2="{start_x + panel_w/2 + 80}" y2="{H - 30}" stroke="#dc2626" stroke-width="2" stroke-dasharray="4,2"/>')
    svg.append(f'<text x="{start_x + panel_w/2 + 90}" y="{H - 26}" font-family="DejaVu Sans, Arial, sans-serif" font-size="12" fill="#1e293b">ZCR</text>')
    
    svg.append('</svg>')
    export_svg_png("\n".join(svg), "curvas_rms_zcr_segmentacao")

# -----------------------------------------------------------------------------
# 3. FIGURA: DISTRIBUIÇÃO DAS CLASSES, LOCUTORES E PARTIÇÕES
# -----------------------------------------------------------------------------
def plot_distributions():
    W = 1100
    H = 650
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    svg.append(f'<text x="{W/2}" y="45" font-family="DejaVu Sans, Arial, sans-serif" font-size="22" font-weight="bold" text-anchor="middle" fill="#1e293b">Distribuição Amostral do Conjunto de Dados MRNA</text>')
    svg.append(f'<text x="{W/2}" y="70" font-family="DejaVu Sans, Arial, sans-serif" font-size="14" fill="#64748b" text-anchor="middle">Contagem de classes silábicas e particionamento experimental por locutor (Total = 504 sílabas)</text>')
    
    # Subgráfico 1: Classes silábicas (DI, REI, TA, ES, QUER, DA)
    # Subgráfico 2: Particionamento (Treino, Validação, Teste)
    # Painel 1 (Classes)
    p1_x, p1_y, p1_w, p1_h = 70, 110, 450, 450
    svg.append(f'<rect x="{p1_x}" y="{p1_y}" width="{p1_w}" height="{p1_h}" fill="#f8fafc" stroke="#cbd5e1" rx="4"/>')
    svg.append(f'<text x="{p1_x + p1_w/2}" y="{p1_y + 30}" font-family="DejaVu Sans, Arial, sans-serif" font-size="16" font-weight="bold" fill="#0f172a" text-anchor="middle">Segmentos por Classe Silábica</text>')
    
    classes = [("DA", 73, 6, 6), ("DI", 68, 8, 7), ("ES", 73, 6, 6), ("QUER", 73, 6, 6), ("REI", 68, 8, 7), ("TA", 68, 8, 7)]
    max_val = 100
    bar_w = 45
    gap = 25
    x_start = p1_x + 40
    
    col_l1, col_l2, col_l3 = "#3b82f6", "#f59e0b", "#10b981"
    
    for idx, (cname, l1, l2, l3) in enumerate(classes):
        bx = x_start + idx * (bar_w + gap)
        h1 = (l1 / max_val) * 320
        h2 = (l2 / max_val) * 320
        h3 = (l3 / max_val) * 320
        y_base = p1_y + 400
        
        y1 = y_base - h1
        y2 = y1 - h2
        y3 = y2 - h3
        
        svg.append(f'<rect x="{bx}" y="{y1:.1f}" width="{bar_w}" height="{h1:.1f}" fill="{col_l1}"/>')
        svg.append(f'<rect x="{bx}" y="{y2:.1f}" width="{bar_w}" height="{h2:.1f}" fill="{col_l2}"/>')
        svg.append(f'<rect x="{bx}" y="{y3:.1f}" width="{bar_w}" height="{h3:.1f}" fill="{col_l3}"/>')
        
        tot = l1 + l2 + l3
        svg.append(f'<text x="{bx + bar_w/2}" y="{y3 - 8:.1f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="12" font-weight="bold" fill="#1e293b" text-anchor="middle">{tot}</text>')
        svg.append(f'<text x="{bx + bar_w/2}" y="{y_base + 20}" font-family="DejaVu Sans, Arial, sans-serif" font-size="12" font-weight="bold" fill="#334155" text-anchor="middle">{cname}</text>')
        
    # Painel 2 (Particionamento)
    p2_x, p2_y, p2_w, p2_h = 580, 110, 450, 450
    svg.append(f'<rect x="{p2_x}" y="{p2_y}" width="{p2_w}" height="{p2_h}" fill="#f8fafc" stroke="#cbd5e1" rx="4"/>')
    svg.append(f'<text x="{p2_x + p2_w/2}" y="{p2_y + 30}" font-family="DejaVu Sans, Arial, sans-serif" font-size="16" font-weight="bold" fill="#0f172a" text-anchor="middle">Partições Experimentais</text>')
    
    parts = [("Treinamento", 318, 30, 27), ("Validação", 63, 6, 6), ("Teste", 42, 6, 6)]
    max_part = 420
    bar_pw = 70
    gap_p = 50
    xp_start = p2_x + 55
    
    for idx, (pname, l1, l2, l3) in enumerate(parts):
        bx = xp_start + idx * (bar_pw + gap_p)
        h1 = (l1 / max_part) * 320
        h2 = (l2 / max_part) * 320
        h3 = (l3 / max_part) * 320
        y_base = p2_y + 400
        
        y1 = y_base - h1
        y2 = y1 - h2
        y3 = y2 - h3
        
        svg.append(f'<rect x="{bx}" y="{y1:.1f}" width="{bar_pw}" height="{h1:.1f}" fill="{col_l1}"/>')
        svg.append(f'<rect x="{bx}" y="{y2:.1f}" width="{bar_pw}" height="{h2:.1f}" fill="{col_l2}"/>')
        svg.append(f'<rect x="{bx}" y="{y3:.1f}" width="{bar_pw}" height="{h3:.1f}" fill="{col_l3}"/>')
        
        tot = l1 + l2 + l3
        svg.append(f'<text x="{bx + bar_pw/2}" y="{y3 - 8:.1f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="12" font-weight="bold" fill="#1e293b" text-anchor="middle">{tot}</text>')
        svg.append(f'<text x="{bx + bar_pw/2}" y="{y_base + 20}" font-family="DejaVu Sans, Arial, sans-serif" font-size="12" font-weight="bold" fill="#334155" text-anchor="middle">{pname}</text>')

    # Legenda comum inferior
    svg.append(f'<rect x="{W/2 - 200}" y="{H - 45}" width="400" height="32" fill="#ffffff" stroke="#cbd5e1" rx="4"/>')
    svg.append(f'<rect x="{W/2 - 180}" y="{H - 34}" width="14" height="14" fill="{col_l1}"/>')
    svg.append(f'<text x="{W/2 - 160}" y="{H - 23}" font-family="DejaVu Sans, Arial, sans-serif" font-size="12" fill="#1e293b">Locutor 1 (423)</text>')
    svg.append(f'<rect x="{W/2 - 50}" y="{H - 34}" width="14" height="14" fill="{col_l2}"/>')
    svg.append(f'<text x="{W/2 - 30}" y="{H - 23}" font-family="DejaVu Sans, Arial, sans-serif" font-size="12" fill="#1e293b">Locutor 2 (42)</text>')
    svg.append(f'<rect x="{W/2 + 80}" y="{H - 34}" width="14" height="14" fill="{col_l3}"/>')
    svg.append(f'<text x="{W/2 + 100}" y="{H - 23}" font-family="DejaVu Sans, Arial, sans-serif" font-size="12" fill="#1e293b">Locutor 3 (39)</text>')

    svg.append('</svg>')
    export_svg_png("\n".join(svg), "distribuicao_amostral_mrna")

# -----------------------------------------------------------------------------
# 4. FIGURA: COMPARAÇÃO ACÚSTICA ENTRE OS LOCUTORES E DURAÇÃO SILÁBICA
# -----------------------------------------------------------------------------
def plot_acoustic_comparison():
    W = 1100
    H = 600
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    svg.append(f'<text x="{W/2}" y="45" font-family="DejaVu Sans, Arial, sans-serif" font-size="22" font-weight="bold" text-anchor="middle" fill="#1e293b">Comparativo Acústico e Métricas Temporais entre Locutores</text>')
    svg.append(f'<text x="{W/2}" y="70" font-family="DejaVu Sans, Arial, sans-serif" font-size="14" fill="#64748b" text-anchor="middle">Duração silábica média, RMS de pico e Taxa de Cruzamento por Zero (ZCR)</text>')
    
    # Métricas agregadas por locutor
    with open("dados/dados_mrna/metadados/silabas_mrna.csv") as f:
        rows = list(csv.DictReader(f))
        
    stats_loc = {}
    for r in rows:
        loc = r["locutor"]
        if loc not in stats_loc:
            stats_loc[loc] = {"dur": [], "rms": [], "zcr": []}
        stats_loc[loc]["dur"].append(float(r["duracao_s"]))
        stats_loc[loc]["rms"].append(float(r["rms_pico_db"]))
        stats_loc[loc]["zcr"].append(float(r["zcr_medio"]))
        
    p_w = 280
    p_h = 420
    col_l = {"locutor_1": "#3b82f6", "locutor_2": "#f59e0b", "locutor_3": "#10b981"}
    
    # 3 Painéis: Duração Média, RMS Pico Médio, ZCR Médio
    panels = [
        ("Duração Silábica Média (s)", "dur", 0.35, 80),
        ("RMS de Pico Médio (dBFS)", "rms", -40.0, 410),
        ("ZCR Médio do Segmento", "zcr", 0.25, 740)
    ]
    
    for title, metric, max_scale, px in panels:
        svg.append(f'<rect x="{px}" y="110" width="{p_w}" height="{p_h}" fill="#f8fafc" stroke="#cbd5e1" rx="4"/>')
        svg.append(f'<text x="{px + p_w/2}" y="140" font-family="DejaVu Sans, Arial, sans-serif" font-size="14" font-weight="bold" fill="#0f172a" text-anchor="middle">{title}</text>')
        
        y_base = 110 + 360
        b_w = 50
        gap = 30
        
        for idx, loc in enumerate(["locutor_1", "locutor_2", "locutor_3"]):
            vals = stats_loc[loc][metric]
            m_val = sum(vals) / len(vals)
            bx = px + 35 + idx * (b_w + gap)
            
            if metric == "rms":
                # Escala de -40 dB a 0 dB
                h = ((m_val - (-40.0)) / (0 - (-40.0))) * 280
                h = max(10, min(280, h))
                y_bar = y_base - h
                val_str = f"{m_val:.1f} dB"
            else:
                h = (m_val / max_scale) * 280
                y_bar = y_base - h
                val_str = f"{m_val:.3f}s" if metric == "dur" else f"{m_val:.3f}"
                
            svg.append(f'<rect x="{bx}" y="{y_bar:.1f}" width="{b_w}" height="{h:.1f}" fill="{col_l[loc]}" rx="2"/>')
            svg.append(f'<text x="{bx + b_w/2}" y="{y_bar - 8:.1f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="11" font-weight="bold" fill="#1e293b" text-anchor="middle">{val_str}</text>')
            svg.append(f'<text x="{bx + b_w/2}" y="{y_base + 20}" font-family="DejaVu Sans, Arial, sans-serif" font-size="11" font-weight="bold" fill="#64748b" text-anchor="middle">Loc {idx+1}</text>')

    svg.append('</svg>')
    export_svg_png("\n".join(svg), "comparativo_acustico_locutores")

# -----------------------------------------------------------------------------
# 5. PAINEL DE INSPEÇÃO VISUAL DOS 27 NOVOS ÁUDIOS (MSWC)
# -----------------------------------------------------------------------------
def plot_inspection_grid():
    # Grid de inspeção compacta dos 27 arquivos (3 colunas x 9 linhas)
    W = 1300
    H = 1750
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    svg.append(f'<text x="{W/2}" y="45" font-family="DejaVu Sans, Arial, sans-serif" font-size="22" font-weight="bold" text-anchor="middle" fill="#1e293b">Painel de Inspeção dos 27 Áudios Multilocutor (MSWC v1.0)</text>')
    svg.append(f'<text x="{W/2}" y="70" font-family="DejaVu Sans, Arial, sans-serif" font-size="14" fill="#64748b" text-anchor="middle">Auditoria visual de envoltória e limites silábicos de cada gravação convertida</text>')
    
    with open("dados/dados_mrna/metadados/silabas_mrna.csv") as f:
        sil_rows = [r for r in csv.DictReader(f) if r["locutor"] in ("locutor_2", "locutor_3")]
        
    # Agrupar por pronuncia
    by_pron = {}
    for r in sil_rows:
        pid = r["id_pronuncia"]
        by_pron.setdefault(pid, []).append(r)
        
    n_cols = 3
    n_rows = 9
    card_w = 360
    card_h = 160
    gap_x = 35
    gap_y = 20
    start_x = 65
    start_y = 100
    
    col_bounds = ["#2563eb", "#d97706", "#059669"]
    col_bg_sil = ["#eff6ff", "#fffbeb", "#ecfdf5"]
    
    pids = sorted(by_pron.keys())
    for idx, pid in enumerate(pids):
        col_idx = idx % n_cols
        row_idx = idx // n_cols
        cx = start_x + col_idx * (card_w + gap_x)
        cy = start_y + row_idx * (card_h + gap_y)
        
        items = by_pron[pid]
        word = items[0]["palavra"]
        loc = items[0]["locutor"]
        part = items[0]["conjunto"]
        has_susp = any(it["suspeita"] == "SIM" for it in items)
        
        # Caminho do WAV da palavra
        wav_path = f"dados/dados_mrna/locutores/{loc}/palavras/{word.lower()}/{pid}.wav"
        _, am_norm, sr = ler_wav_pcm(wav_path)
        dur = len(am_norm) / sr
        
        bord_color = "#ef4444" if has_susp else "#cbd5e1"
        bg_card = "#fff1f2" if has_susp else "#f8fafc"
        
        svg.append(f'<rect x="{cx}" y="{cy}" width="{card_w}" height="{card_h}" fill="{bg_card}" stroke="{bord_color}" stroke-width="{1.8 if has_susp else 1}" rx="4"/>')
        
        # Cabeçalho do card
        tag_susp = " [SUSPEITO]" if has_susp else ""
        svg.append(f'<text x="{cx + 10}" y="{cy + 20}" font-family="DejaVu Sans, Arial, sans-serif" font-size="12" font-weight="bold" fill="#0f172a">{pid} ({word}) - {part.upper()}{tag_susp}</text>')
        
        # Waveform interior
        plot_y = cy + 30
        plot_h = card_h - 45
        mid_y = plot_y + plot_h / 2.0
        scale_y = (plot_h - 10) / 2.0
        
        # Fatias das sílabas
        for it, col, bg in zip(items, col_bounds, col_bg_sil):
            b_ini = float(it["inicio_s"])
            b_fim = float(it["fim_s"])
            sname = it["classe"]
            x1 = cx + (b_ini / dur) * card_w
            x2 = cx + (b_fim / dur) * card_w
            svg.append(f'<rect x="{x1:.1f}" y="{plot_y}" width="{(x2-x1):.1f}" height="{plot_h}" fill="{bg}" opacity="0.6"/>')
            svg.append(f'<line x1="{x1:.1f}" y1="{plot_y}" x2="{x1:.1f}" y2="{plot_y+plot_h}" stroke="{col}" stroke-width="1.2" stroke-dasharray="2,2"/>')
            svg.append(f'<line x1="{x2:.1f}" y1="{plot_y}" x2="{x2:.1f}" y2="{plot_y+plot_h}" stroke="{col}" stroke-width="1.2" stroke-dasharray="2,2"/>')
            svg.append(f'<text x="{(x1+x2)/2:.1f}" y="{plot_y + 14}" font-family="DejaVu Sans, Arial, sans-serif" font-size="10" font-weight="bold" fill="{col}" text-anchor="middle">{sname}</text>')
            
        step = max(1, len(am_norm) // 250)
        d_path = []
        for s_i in range(0, len(am_norm), step):
            chunk = am_norm[s_i:s_i+step]
            t_s = (s_i + len(chunk)/2) / sr
            x = cx + (t_s / dur) * card_w
            y_max = mid_y - max(chunk) * scale_y
            y_min = mid_y - min(chunk) * scale_y
            d_path.append(f"M {x:.1f} {y_min:.1f} L {x:.1f} {y_max:.1f}")
        svg.append(f'<path d="{" ".join(d_path)}" stroke="#334155" stroke-width="0.9"/>')

    svg.append('</svg>')
    export_svg_png("\n".join(svg), "painel_inspecao_27_audios_mrna")

if __name__ == "__main__":
    plot_waveform_examples()
    plot_rms_zcr_curves()
    plot_distributions()
    plot_acoustic_comparison()
    plot_inspection_grid()
    print("Todas as figuras foram geradas e convertidas com sucesso!")
