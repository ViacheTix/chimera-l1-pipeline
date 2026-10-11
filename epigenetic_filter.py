"""
epigenetic_filter.py

Применяет эпигенетический фильтр нулевого уровня (M_active) к интактным копиям LINE-1 (L1HS/L1PA1).
Оценка активности транскрипции (GRO-seq) и репрессии (H3K9me3) производится СТРОГО в области 
внутреннего промотора (5'-UTR, первые 900 нт).
Для активных локусов извлекается домен ORF2 для последующего поиска мишеней микроРНК.
"""

import argparse
import logging
import math
import os
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import pyBigWig
import pybedtools
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("epigenetic_filter")

# Биологические константы
MIN_INTACT_L1_LENGTH = 5800  # Строгий порог длины для полноразмерных элементов
UTR5_LENGTH = 900            # Длина 5'-UTR (промоторная зона для оценки эпигенетики)
ORF2_5P_OFFSET = 1980        # Начало ORF2 относительно 5'-TSS
ORF2_3P_OFFSET = 5810        # Конец ORF2 относительно 5'-TSS
MAX_CPU_WORKERS = 24         # Лимит потоков (оставляем 4 для стабильности ОС)


def parse_l1_bed(bed_path: Path) -> pd.DataFrame:
    """Парсинг и строгая биологическая фильтрация BED-файла LINE-1."""
    if not bed_path.exists():
        raise FileNotFoundError(f"Входной BED файл не найден: {bed_path}")

    col_names = ["chrom", "start", "end", "name", "score", "strand"]
    df = pd.read_csv(bed_path, sep=r"\s+", header=None, names=col_names, engine="python", comment="#")
    
    # 1. Фильтрация канонических хромосом
    df["chrom"] = df["chrom"].astype(str)
    df["chrom"] = df["chrom"].apply(lambda c: c if c.startswith("chr") else f"chr{c}")
    df = df[df["chrom"].str.match(r"^chr([1-9]|1[0-9]|2[0-2]|X|Y)$")].copy()

    # 2. Фильтрация по эволюционным подсемействам (только молодые)
    initial_len = len(df)
    target_subfamilies = {"L1HS", "L1PA1"}
    df = df[df["name"].isin(target_subfamilies)].copy()
    dropped_subf = initial_len - len(df)
    if dropped_subf > 0:
        logger.info(f"Отфильтровано {dropped_subf} локусов старых подсемейств. Оставлены только L1HS/L1PA1.")

    # 3. Фильтрация усеченных элементов
    current_len = len(df)
    df = df[(df["end"] - df["start"]) >= MIN_INTACT_L1_LENGTH].copy()
    dropped_trunc = current_len - len(df)
    if dropped_trunc > 0:
        logger.info(f"Отфильтровано {dropped_trunc} усеченных локусов (< {MIN_INTACT_L1_LENGTH} bp).")

    df["locus_id"] = df["name"] + "_" + df["chrom"] + "_" + df["start"].astype(str) + "_" + df["end"].astype(str)
    return df


def filter_host_readthrough(df: pd.DataFrame, gtf_path: Optional[Path]) -> pd.DataFrame:
    """Удаление локусов L1, сонаправленных с транскрибируемыми генами-хозяевами."""
    if not gtf_path or not gtf_path.exists():
        logger.warning("GTF файл не предоставлен. Фильтрация host-gene read-through пропущена.")
        return df

    bed6_df = df[["chrom", "start", "end", "locus_id", "score", "strand"]].copy()
    bed6_df["score"] = bed6_df["score"].fillna(0).astype(int)
    
    l1_bt = pybedtools.BedTool.from_dataframe(bed6_df)
    gtf_bt = pybedtools.BedTool(str(gtf_path))

    inter = l1_bt.intersect(gtf_bt, s=True, u=True)
    overlap_ids = {hit.name for hit in inter}
    pybedtools.cleanup()

    logger.info(f"Обнаружено {len(overlap_ids)} локусов внутри активных генов (со-направленных).")
    return df[~df["locus_id"].isin(overlap_ids)].copy()


def extract_5p_utr_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Извлекает координаты 5'-UTR промотора для точной оценки эпигенетических меток."""
    utr_df = df.copy()
    pos_mask = utr_df["strand"] == "+"
    
    # Для (+) цепи: от start до start + UTR5_LENGTH
    utr_df.loc[pos_mask, "end"] = utr_df.loc[pos_mask, "start"] + UTR5_LENGTH
    # Для (-) цепи: от end - UTR5_LENGTH до end
    utr_df.loc[~pos_mask, "start"] = utr_df.loc[~pos_mask, "end"] - UTR5_LENGTH
    
    return utr_df


def _worker_query_bigwig(bw_path_str: str, intervals: List[Tuple[str, int, int]], invert_sign: bool = False) -> List[float]:
    """Воркер для параллельного запроса среднего сигнала в BigWig."""
    results = []
    try:
        with pyBigWig.open(bw_path_str) as bw:
            bw_chroms = set(bw.chroms().keys())
            for chrom, start, end in intervals:
                target_chrom = chrom if chrom in bw_chroms else chrom.replace("chr", "")
                if target_chrom not in bw_chroms:
                    results.append(0.0)
                    continue
                
                chrom_len = bw.chroms(target_chrom)
                s_clamped = max(0, min(int(start), chrom_len - 1))
                e_clamped = max(s_clamped + 1, min(int(end), chrom_len))
                
                try:
                    val = bw.stats(target_chrom, s_clamped, e_clamped, type="mean")
                    if val and val[0] is not None and not math.isnan(val[0]):
                        raw_val = float(val[0])
                        processed_val = abs(raw_val) if invert_sign else max(0.0, raw_val)
                        results.append(processed_val)
                    else:
                        results.append(0.0)
                except Exception:
                    results.append(0.0)
    except Exception as exc:
        logger.error(f"Ошибка чтения BigWig {bw_path_str}: {exc}")
        return [0.0] * len(intervals)
    return results


def compute_bigwig_signals(df: pd.DataFrame, bigwig_path: Path, desc: str, n_jobs: int, invert_sign: bool = False) -> pd.Series:
    """Многопроцессорный расчет сигнала. Использует координаты из df (предполагается 5'-UTR)."""
    intervals = list(zip(df["chrom"], df["start"], df["end"]))
    if not intervals:
        return pd.Series(0.0, index=df.index, dtype=np.float64)

    n_splits = max(1, min(n_jobs, len(intervals)))
    chunks = [c.tolist() for c in np.array_split(intervals, n_splits) if len(c) > 0]
    bw_path_str = str(bigwig_path.resolve())
    
    results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
        joblib.delayed(_worker_query_bigwig)(bw_path_str, chunk, invert_sign)
        for chunk in tqdm(chunks, desc=f"Quantifying {desc}", leave=False)
    )

    flattened = [val for sublist in results for val in sublist]
    return pd.Series(flattened, index=df.index, dtype=np.float64)


def isolate_orf2(df: pd.DataFrame) -> pd.DataFrame:
    """Извлечение координат домена ORF2 для поиска мишеней микроРНК."""
    out_df = df.copy()
    pos_mask = (out_df["strand"] == "+").values
    orig_start = out_df["start"].values
    orig_end = out_df["end"].values
    
    pos_s = orig_start + ORF2_5P_OFFSET
    pos_e = np.minimum(orig_start + ORF2_3P_OFFSET, orig_end)
    
    neg_s = np.maximum(orig_end - ORF2_3P_OFFSET, orig_start)
    neg_e = orig_end - ORF2_5P_OFFSET
    
    out_df["start"] = np.where(pos_mask, pos_s, neg_s).astype(int)
    out_df["end"] = np.where(pos_mask, pos_e, neg_e).astype(int)
    
    valid_coords = (out_df["start"] < out_df["end"]) & ((out_df["end"] - out_df["start"]) >= 3500)
    invalid_count = (~valid_coords).sum()
    if invalid_count > 0:
        logger.warning(f"Удалено {invalid_count} локусов со слишком коротким ORF2.")
        
    return out_df[valid_coords].copy()


def plot_epigenetic_segregation(df: pd.DataFrame, tau_gro: float, tau_h3: float, out_path: Path):
    """Построение scatter plot для визуализации распределения сигналов 5'-UTR."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    sns.set_theme(style="whitegrid", context="paper")
    
    df["status"] = df["M_active"].map({0: "TGS-repressed", 1: "Active / PTGS-eligible"})
    
    sns.scatterplot(
        data=df, x="S_GRO", y="S_H3", hue="status",
        palette={"TGS-repressed": "#D62728", "Active / PTGS-eligible": "#2CA02C"},
        alpha=0.6, s=40, edgecolor=None
    )
    
    plt.axvline(x=tau_gro, color="#1F77B4", linestyle="--", lw=2, label=f"$\\tau_{{nascent}}$ = {tau_gro:.2f}")
    plt.axhline(y=tau_h3, color="#FF7F0E", linestyle="--", lw=2, label=f"$\\tau_{{H3}}$ = {tau_h3:.2f}")
    
    plt.xlabel("Nascent Transcription in 5'-UTR ($S_{GRO}$)", fontsize=12)
    plt.ylabel("H3K9me3 Heterochromatin in 5'-UTR ($S_{H3}$)", fontsize=12)
    plt.title("Epigenetic Zero-Level Gate (Promoter Evaluation)", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", dpi=300)
    plt.close()
    logger.info(f"График сохранен: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Epigenetic Filter (Promoter Localization) & ORF2 Isolation")
    parser.add_argument("--bed", type=Path, default=Path("data/raw/genome/L1_filtered.bed"), help="Path to intact L1 BED")
    parser.add_argument("--h3-bigwig", type=Path, default=Path("data/raw/epigenome/ENCFF066MEE.bigWig"), help="H3K9me3 track")
    parser.add_argument("--gro-plus", type=Path, default=Path("data/raw/epigenome/ENCFF421TGH_plus.bigWig"), help="Nascent plus track")
    parser.add_argument("--gro-minus", type=Path, default=Path("data/raw/epigenome/ENCFF554WDC_minus.bigWig"), help="Nascent minus track")
    parser.add_argument("--gtf", type=Path, default=Path("data/raw/genome/gencode.v45.annotation.gtf.gz"), help="Host genes GTF")
    
    # Биологически обоснованные пороги по умолчанию: 
    # Транскрипция > 0.05 (минимально значимая), Гетереохроматин FC > 1.0 (репрессия)
    parser.add_argument("--tau-gro", type=float, default=0.05, help="Threshold for nascent transcription")
    parser.add_argument("--tau-h3", type=float, default=1.0, help="Threshold for H3K9me3")
    
    parser.add_argument("--n-jobs", type=int, default=MAX_CPU_WORKERS, help="Number of CPU threads")
    parser.add_argument("--out-bed", type=Path, default=Path("data/prepped/active_orf2.bed"), help="Output BED path")
    parser.add_argument("--out-plot", type=Path, default=Path("data/figures/epigenetic_scatter.pdf"), help="Output plot path")

    args = parser.parse_args()
    workers = min(args.n_jobs, MAX_CPU_WORKERS)
    
    logger.info("Запуск модуля epigenetic_filter...")
    
    # 1. Парсинг L1 локусов и удаление генов-хозяев
    df = parse_l1_bed(args.bed)
    df = filter_host_readthrough(df, args.gtf)
    
    # 2. Локализация сигнала (изолируем 5'-UTR для оценки BigWig)
    utr5_df = extract_5p_utr_coords(df)
    
    # 3. Экстракция сигналов в зоне 5'-UTR
    logger.info("Расчет H3K9me3 сигнала в зоне 5'-UTR...")
    df["S_H3"] = compute_bigwig_signals(utr5_df, args.h3_bigwig, "H3K9me3", workers)
    
    logger.info("Расчет Nascent-сигнала в зоне 5'-UTR...")
    df["S_GRO"] = 0.0
    plus_mask = df["strand"] == "+"
    minus_mask = df["strand"] == "-"
    
    if plus_mask.any():
        df.loc[plus_mask, "S_GRO"] = compute_bigwig_signals(utr5_df[plus_mask], args.gro_plus, "Nascent (+)", workers)
    if minus_mask.any():
        df.loc[minus_mask, "S_GRO"] = compute_bigwig_signals(utr5_df[minus_mask], args.gro_minus, "Nascent (-)", workers, invert_sign=True)

    # 4. Применение маски M_active
    df["M_active"] = ((df["S_GRO"] > args.tau_gro) & (df["S_H3"] < args.tau_h3)).astype(int)
    active_df = df[df["M_active"] == 1].copy()
    logger.info(f"Активных локусов после оценки промоторов: {len(active_df)} из {len(df)}")
    
    if active_df.empty:
        raise RuntimeError("Ни один локус не прошел эпигенетический фильтр. Проверьте данные или пороги.")

    # 5. Выделение ORF2 из оригинальных полноразмерных координат
    orf2_df = isolate_orf2(active_df)
    
    # Сохранение результатов
    args.out_bed.parent.mkdir(parents=True, exist_ok=True)
    bed6_cols = ["chrom", "start", "end", "locus_id", "M_active", "strand"]
    orf2_df[bed6_cols].to_csv(args.out_bed, sep="\t", header=False, index=False)
    logger.info(f"Сохранен файл активных ORF2: {args.out_bed}")

    # 6. Визуализация
    plot_epigenetic_segregation(df, args.tau_gro, args.tau_h3, args.out_plot)
    logger.info("Эпигенетический фильтр успешно завершен.")

if __name__ == "__main__":
    main()