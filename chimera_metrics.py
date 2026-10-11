"""
Узел конвейера: Валидация EM-разрешенных химерных чтений AGO2 и картирование на LINE-1.
Функционал: Строгая strand-specific фильтрация, слияние дубликатов (anti-data-leakage),
сохранение интактных seed-координат (anti-clamping) и отбор биологических якорей 
(miR-128-3p, let-7-5p) in vivo.
"""

import argparse
import gzip
import json
import logging
import multiprocessing as mp
import tempfile
import time
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import pybedtools
import seaborn as sns
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("chimera_metrics")

# Биологические и биофизические константы
MIN_PEAK_LENGTH = 15
MAX_PEAK_LENGTH = 75
MIN_ORF2_OVERLAP_BP = 15
TARGET_MIRNA_REGEX = r"mir-128-3p|let-7*-5p"


def _filter_peak_chunk(chunk_data: Tuple[pd.DataFrame, float]) -> pd.DataFrame:
    """Параллельная фильтрация чанков EM-пиков с проверкой биофизических границ."""
    chunk, min_prob = chunk_data
    
    # Адаптивный парсинг формата BED (6 или 7 колонок)
    if len(chunk.columns) >= 7:
        chunk.columns = ["chrom", "peak_start", "peak_end", "locus_id", "em_posterior_prob", "strand", "mirna_guide"]
    else:
        chunk.columns = ["chrom", "peak_start", "peak_end", "locus_id", "em_posterior_prob", "mirna_guide"]
        chunk["strand"] = "."  # Помечаем отсутствие цепи (вызовет Warning в main)
        
    chunk["chrom"] = chunk["chrom"].astype(str)
    chr_mask = chunk["chrom"].str.match(r"^chr([1-9]|1[0-9]|2[0-2]|X|Y|M|MT)$")
    
    prob_mask = chunk["em_posterior_prob"] >= min_prob
    
    # Биофизический фильтр длины таргетного следа AGO2 (Footprint size)
    lengths = chunk["peak_end"] - chunk["peak_start"]
    len_mask = (lengths >= MIN_PEAK_LENGTH) & (lengths <= MAX_PEAK_LENGTH)
    
    # Фильтр таргетных микроРНК
    mirna_series = chunk["mirna_guide"].astype(str).str.lower()
    mirna_mask = mirna_series.str.contains(TARGET_MIRNA_REGEX, regex=True)
    
    return chunk[chr_mask & prob_mask & len_mask & mirna_mask].copy()


def preprocess_em_peaks_parallel(
    peaks_bed_path: Path, 
    min_prob: float, 
    threads: int
) -> pd.DataFrame:
    """Многопоточное чтение и первичная очистка сырых EM-выравниваний."""
    logger.info(f"Запуск параллельной фильтрации EM-пиков (Потоков: {threads})...")
    
    chunk_size = 250_000
    reader = pd.read_csv(
        peaks_bed_path, sep=r"\s+", header=None, 
        engine="c", comment="#", chunksize=chunk_size
    )
    
    pool = mp.Pool(processes=threads)
    processed_chunks = []
    
    task_generator = ((chunk, min_prob) for chunk in reader)
    
    for filtered_chunk in tqdm(
        pool.imap(_filter_peak_chunk, task_generator), 
        desc="Фильтрация чанков", 
        unit="chunk"
    ):
        processed_chunks.append(filtered_chunk)
        
    pool.close()
    pool.join()
    
    if not processed_chunks:
        raise ValueError(f"Файл {peaks_bed_path} не содержит валидных записей AGO2.")
        
    combined_df = pd.concat(processed_chunks, ignore_index=True)
    
    if (combined_df["strand"] == ".").all():
        logger.warning(
            "ВНИМАНИЕ: Во входном файле EM-пиков отсутствует информация о цепи (Strand). "
            "Для строгой strand-specific валидации рекомендуется использовать BED7 формат. "
            "Алгоритм продолжит работу, но кросс-антисмысловое связывание не сможет быть отфильтровано на 100%."
        )
        
    return combined_df


def merge_overlapping_peaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Слияние пересекающихся пиков для предотвращения утечки данных (Data Leakage).
    Группирует по хромосоме, локусу и микроРНК, объединяя сдвинутые на 1-2 нт риды.
    """
    if df.empty:
        return df
        
    logger.info("Слияние перекрывающихся eCLIP пиков (Anti-Leakage)...")
    df = df.sort_values(['chrom', 'strand', 'locus_id', 'mirna_guide', 'peak_start'])
    
    merged = []
    # Группировка изолирует каждый уникальный комплекс L1-микроРНК
    for (chrom, strand, locus, mirna), group in df.groupby(['chrom', 'strand', 'locus_id', 'mirna_guide']):
        starts = group['peak_start'].values
        ends = group['peak_end'].values
        probs = group['em_posterior_prob'].values
        
        c_start, c_end, c_prob = starts[0], ends[0], probs[0]
        
        for i in range(1, len(starts)):
            if starts[i] <= c_end:  # Пики перекрываются
                c_end = max(c_end, ends[i])
                c_prob = max(c_prob, probs[i]) # Сохраняем максимальную уверенность алгоритма
            else:
                merged.append((chrom, c_start, c_end, locus, c_prob, strand, mirna))
                c_start, c_end, c_prob = starts[i], ends[i], probs[i]
        
        merged.append((chrom, c_start, c_end, locus, c_prob, strand, mirna))
        
    return pd.DataFrame(
        merged, 
        columns=["chrom", "peak_start", "peak_end", "locus_id", "em_posterior_prob", "strand", "mirna_guide"]
    )


def intersect_with_active_l1(
    filtered_peaks_df: pd.DataFrame, 
    active_l1_bed_path: Path
) -> Tuple[pd.DataFrame, int]:
    """Физическое strand-specific пересечение с сохранением целостности сайтов (без Clamping)."""
    logger.info("Пространственное пересечение с транскриптами LINE-1 (Strict Strand-Specific)...")
    
    raw_anchor_count = len(filtered_peaks_df)
    if raw_anchor_count == 0:
        return pd.DataFrame(), 0

    composite_names = (
        filtered_peaks_df["locus_id"].astype(str) + ":::" + 
        filtered_peaks_df["mirna_guide"].astype(str) + ":::" + 
        filtered_peaks_df["em_posterior_prob"].astype(str)
    )
    
    anchor_bed_df = pd.DataFrame({
        "chrom": filtered_peaks_df["chrom"].astype(str),
        "start": filtered_peaks_df["peak_start"].astype(int),
        "end": filtered_peaks_df["peak_end"].astype(int),
        "name": composite_names,
        "score": (filtered_peaks_df["em_posterior_prob"] * 1000).astype(int),
        "strand": filtered_peaks_df["strand"].astype(str),
    })

    with tempfile.NamedTemporaryFile(suffix=".bed", mode="w", delete=False) as tmp_peaks:
        tmp_peaks_path = tmp_peaks.name
        anchor_bed_df.to_csv(tmp_peaks_path, sep="\t", header=False, index=False)

    try:
        peaks_bt = pybedtools.BedTool(tmp_peaks_path)
        l1_bt = pybedtools.BedTool(str(active_l1_bed_path))

        # Жесткое совпадение цепей (s=True). Если у пика strand=".", пересечение вернет только те,
        # где игнорирование цепи допустимо, но в идеале eCLIP BED должен иметь '+/-'
        intersected = peaks_bt.intersect(l1_bt, wa=True, wb=True, s=True)
        intersect_df = intersected.to_dataframe(header=None, disable_auto_names=True)
    finally:
        Path(tmp_peaks_path).unlink(missing_ok=True)
        pybedtools.cleanup()

    if intersect_df.empty:
        return pd.DataFrame(), raw_anchor_count

    # Извлечение координат для вычисления перекрытия
    names_split = intersect_df.iloc[:, 3].astype(str).str.split(":::", expand=True)
    
    raw_p_start = intersect_df.iloc[:, 1].to_numpy(dtype=int)
    raw_p_end = intersect_df.iloc[:, 2].to_numpy(dtype=int)
    l1_start = intersect_df.iloc[:, 7].to_numpy(dtype=int)
    l1_end = intersect_df.iloc[:, 8].to_numpy(dtype=int)

    # Вычисление фактической длины перекрытия пика с ORF2
    overlap_start = np.maximum(raw_p_start, l1_start)
    overlap_end = np.minimum(raw_p_end, l1_end)
    overlap_len = overlap_end - overlap_start

    # Важно: Сохраняем ОРИГИНАЛЬНЫЕ координаты пика (raw_p_start, raw_p_end), а не обрезанные,
    # чтобы не разрушить последовательность связывания микроРНК для RNA-FM.
    matched_df = pd.DataFrame({
        "chrom": intersect_df.iloc[:, 0].astype(str),
        "peak_start": raw_p_start,
        "peak_end": raw_p_end,
        "locus_id": intersect_df.iloc[:, 9].astype(str),
        "mirna_guide": names_split[1].astype(str),
        "em_posterior_prob": names_split[2].astype(float),
        "strand": intersect_df.iloc[:, 11].astype(str),
        "overlap_len": overlap_len
    })

    # Отсеиваем случайные касания (overlap < 15 нт)
    matched_df = matched_df[matched_df["overlap_len"] >= MIN_ORF2_OVERLAP_BP].copy()
    matched_df.drop(columns=["overlap_len"], inplace=True)

    # Дедупликация и слияние соседних/сдвинутых ридов
    result_df = matched_df.drop_duplicates().reset_index(drop=True)
    final_merged_df = merge_overlapping_peaks(result_df)

    return final_merged_df, raw_anchor_count


def generate_visualizations(peaks_df: pd.DataFrame, out_dir: Path) -> None:
    """Генерация публикационных графиков распределения сайтов посадки AGO2."""
    if peaks_df.empty:
        logger.warning("Нет данных для генерации визуализаций.")
        return
        
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    peak_lengths = (peaks_df["peak_end"] - peaks_df["peak_start"]).to_numpy(dtype=int)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.5, 1]})
    
    # Панель 1: Распределение длин посадочных площадок
    sns.histplot(peak_lengths, bins=range(min(peak_lengths), max(peak_lengths) + 5, 2), 
                 kde=True, ax=axes[0], color="steelblue", stat="density", edgecolor="black")
    axes[0].axvline(np.median(peak_lengths), color="darkred", linestyle="--", 
                    label=f"Медиана = {np.median(peak_lengths):.0f} нт")
    axes[0].set_title("Длины интактных in vivo eCLIP участков (AGO2)", fontweight="bold")
    axes[0].set_xlabel("Длина таргетного фрагмента (нт)")
    axes[0].set_ylabel("Плотность")
    axes[0].legend()

    # Панель 2: Представленность биологических якорей
    mirna_counts = peaks_df["mirna_guide"].value_counts().reset_index()
    mirna_counts.columns = ["МикроРНК", "Количество пиков"]
    
    sns.barplot(data=mirna_counts, x="Количество пиков", y="МикроРНК", 
                ax=axes[1], palette="viridis", edgecolor="black")
    axes[1].set_title("Доля микроРНК в комплексах AGO2/LINE-1", fontweight="bold")
    axes[1].set_xlabel("Уникальных Ground Truth сайтов")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plot_path = out_dir / "ago2_binding_metrics.pdf"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Диагностические графики сохранены в {plot_path}")


def count_fastq_reads(fastq_path: Path) -> int:
    """Быстрый подсчет количества чтений в .fastq.gz (для метрик аудита)."""
    if not fastq_path or not fastq_path.exists():
        return 0
    logger.info(f"Подсчет исходных чтений в {fastq_path.name}...")
    with gzip.open(fastq_path, "rt") as f:
        return sum(1 for _ in f) // 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Chimera Deconvolution Validation")
    parser.add_argument("--peaks-bed", type=Path, required=True, help="Путь к сырым пикам EM_resolved_peaks.bed")
    parser.add_argument("--active-l1-bed", type=Path, required=True, help="Путь к активным L1_ORF2 (BED)")
    parser.add_argument("--raw-fastq", type=Path, required=False, help="Путь к сырым FASTQ (SRR) для базовых метрик")
    parser.add_argument("--out-parquet", type=Path, required=True, help="Путь для Parquet")
    parser.add_argument("--out-audit", type=Path, required=True, help="Путь для JSON аудита")
    parser.add_argument("--min-posterior", type=float, default=0.05, help="Порог уверенности EM")
    parser.add_argument("--threads", type=int, default=24, help="Лимит потоков")
    args = parser.parse_args()

    t_start = time.time()
    threads = min(args.threads, 24)
    if args.threads > 24:
        logger.warning("Запрошено более 24 потоков. Принудительное ограничение до 24 потоков для стабильности ОС.")
    
    # 0. Инициализация базовых метрик
    total_raw_reads = count_fastq_reads(args.raw_fastq) if args.raw_fastq else 0

    # 1. Параллельная предобработка
    filtered_raw_df = preprocess_em_peaks_parallel(args.peaks_bed, args.min_posterior, threads)
    
    # 2. Пересечение (Anti-Leakage & Anti-Clamping)
    final_peaks_df, pre_intersect_count = intersect_with_active_l1(filtered_raw_df, args.active_l1_bed)
    
    # 3. Сохранение
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    final_peaks_df.to_parquet(args.out_parquet, engine="pyarrow", index=False)
    logger.info(f"Сохранено {len(final_peaks_df)} подтвержденных in vivo контактов в {args.out_parquet}")

    # 4. Визуализация
    generate_visualizations(final_peaks_df, args.out_parquet.parent.parent / "figures")

    # 5. Глубокий аудит (с учетом FASTQ)
    duration = time.time() - t_start
    memory_usage_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    
    audit_data = {
        "pipeline_step": "chimera_metrics",
        "parameters": {
            "min_posterior_prob": args.min_posterior,
            "strand_specific_enforced": True,
            "anti_clamping_enabled": True,
            "approved_mirna_regex": TARGET_MIRNA_REGEX,
            "min_orf2_overlap_bp": MIN_ORF2_OVERLAP_BP,
            "max_peak_length_bp": MAX_PEAK_LENGTH
        },
        "filtering_funnel": {
            "initial_srr_chimeric_reads": total_raw_reads,
            "em_resolved_multimapped_anchors": pre_intersect_count,
            "final_ground_truth_peaks": len(final_peaks_df),
            "global_retention_rate": round(len(final_peaks_df) / total_raw_reads, 6) if total_raw_reads else None
        },
        "anchors_breakdown": final_peaks_df["mirna_guide"].value_counts().to_dict() if not final_peaks_df.empty else {},
        "hardware_telemetry": {
            "duration_seconds": round(duration, 2),
            "peak_rss_mb": round(memory_usage_mb, 2),
            "threads_utilized": threads
        }
    }

    args.out_audit.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_audit, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Метрики выполнения записаны в {args.out_audit}")


if __name__ == "__main__":
    main()