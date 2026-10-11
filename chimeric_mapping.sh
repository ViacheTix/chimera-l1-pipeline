#!/usr/bin/env bash
#
# chimeric_mapping.sh
# 
# Этап 2.2: In silico деконволюция химерных чтений и разрешение мульти-маппинга.
# Объединяет надежную архитектуру оркестрации с точной биологической методологией
# (ChiRA для split-mapping и TElocal для EM-алгоритма).

set -euo pipefail

# --- 1. Глобальные ловушки и очистка ресурсов ---
TMP_DIR=""
cleanup() {
    local exit_code=$?
    if [ -n "${TMP_DIR:-}" ] && [ -d "${TMP_DIR}" ]; then
        rm -rf "${TMP_DIR}"
    fi
    if [ "${exit_code}" -ne 0 ]; then
        echo "[ERROR] Пайплайн завершился с ошибкой (код ${exit_code})." >&2
    fi
    exit "${exit_code}"
}
trap cleanup EXIT
trap 'echo "[WARNING] Прервано пользователем." >&2; exit 130' SIGINT SIGTERM
trap 'echo "[ERROR] Ошибка на строке ${LINENO}. Команда: ${BASH_COMMAND}" >&2' ERR

echo "[INFO] ========================================================"
echo "[INFO] Запуск chimeric_mapping.sh"
echo "[INFO] ========================================================"

# --- 2. Настройка среды и аппаратных лимитов ---
mkdir -p data/prepped data/lineage data/raw/genome/star_index logs

# Ограничение для Intel Xeon E5-2680 v4 (оставляем ресурсы для ОС)
THREADS=24
echo "[INFO] Выделено ${THREADS} параллельных потоков (CPU Limit)."

# Поиск Python (аналогично вашему драфту)
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "[ERROR] Python 3 не найден." >&2
    exit 1
fi
echo "[INFO] Используется Python: ${PYTHON_BIN}"

# --- 3. Входные данные ---
# Контрольные данные эксперимента
RAW_R1=("data/raw/eclip/SRR34920805_1.fastq.gz" "data/raw/eclip/SRR34920806_1.fastq.gz")
RAW_R2=("data/raw/eclip/SRR34920805_2.fastq.gz" "data/raw/eclip/SRR34920806_2.fastq.gz")
MIRNA_FA="data/raw/genome/mature.fa"
ACTIVE_L1_BED="data/prepped/active_orf2.bed"

for fq in "${RAW_R1[@]}" "${RAW_R2[@]}"; do
    if [ ! -f "${fq}" ]; then
        echo "[ERROR] Отсутствует FASTQ файл: ${fq}" >&2
        exit 1
    fi
done

if [ ! -f "${ACTIVE_L1_BED}" ]; then
    echo "[ERROR] Отсутствует ${ACTIVE_L1_BED}. Сначала выполните epigenetic_filter.py" >&2
    exit 1
fi

TMP_DIR=$(mktemp -d -t chimera_proc_XXXXXX)
echo "[INFO] Создана временная директория: ${TMP_DIR}"

# --- 4. Конвейер деконволюции и маппинга ---

# 4.1 Cutadapt: Базовая очистка и удаление Illumina адаптеров
echo "[INFO] ШАГ 1: Контроль качества FASTQ (Cutadapt)..."
TRIMMED_R1="${TMP_DIR}/merged_trimmed_R1.fastq.gz"
TRIMMED_R2="${TMP_DIR}/merged_trimmed_R2.fastq.gz"

# Объединяем на лету и чистим
cat "${RAW_R1[@]}" | cutadapt -j "${THREADS}" -q 20 --minimum-length 15 \
    -a AGATCGGAAGAGCACACGTCTGAACTCCAGTCA \
    -o "${TRIMMED_R1}" - \
    > logs/cutadapt_R1.log

cat "${RAW_R2[@]}" | cutadapt -j "${THREADS}" -q 20 --minimum-length 15 \
    -A AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT \
    -o "${TRIMMED_R2}" - \
    > logs/cutadapt_R2.log

# 4.2 ChiRA: Split-mapping
echo "[INFO] ШАГ 2: Извлечение микроРНК и таргетных участков через ChiRA..."
CHIRA_OUT="${TMP_DIR}/chira_out"
mkdir -p "${CHIRA_OUT}"
# ChiRA разделяет рид. Результат: FASTQ с таргетными фрагментами
chira extract \
    -m "${MIRNA_FA}" \
    -1 "${TRIMMED_R1}" -2 "${TRIMMED_R2}" \
    -o "${CHIRA_OUT}" \
    -t "${THREADS}" > logs/chira.log 2>&1

TARGETS_FQ="${CHIRA_OUT}/targets.fastq.gz"
if [ ! -f "${TARGETS_FQ}" ]; then
    echo "[ERROR] ChiRA не сформировал таргетный FASTQ." >&2
    exit 1
fi

# 4.3 STAR: Мульти-маппинг
STAR_IDX="data/indices/STAR_hg38"
OUT_BAM="data/prepped/chimera_targets_multimapped.bam"

echo "[INFO] ШАГ 3: Выравнивание таргетных фрагментов STAR (с сохранением мульти-маппинга)..."
STAR --runThreadN "${THREADS}" \
     --genomeDir "${STAR_IDX}" \
     --readFilesIn "${TARGETS_FQ}" \
     --readFilesCommand "gzip -dc" \
     --outFilterMultimapNmax 100 \
     --winAnchorMultimapNmax 100 \
     --outSAMtype BAM SortedByCoordinate \
     --outFileNamePrefix "${TMP_DIR}/star_"

mv "${TMP_DIR}/star_Aligned.sortedByCoord.out.bam" "${OUT_BAM}"
samtools index -@ "${THREADS}" "${OUT_BAM}"

# 4.4 TElocal: EM-алгоритм
TELOCAL_IDX="data/indices/hg38_rmsk_TE.gtf.locInd.gz"
EM_BAM="${TMP_DIR}/em_resolution.bam"
PEAKS_BED="data/prepped/EM_resolved_peaks.bed"

echo "[INFO] ШАГ 4: Разрешение мульти-маппинга EM-алгоритмом (TElocal)..."
TElocal -b "${OUT_BAM}" \
        --GTF "${TELOCAL_IDX}" \
        --mode multi \
        --project "${TMP_DIR}/em" \
        --sortByPos

# Конвертация BAM с весами EM (posterior probability) в BED7
echo "[INFO] Экстракция пиков EM в BED формат..."
samtools view -@ "${THREADS}" "${TMP_DIR}/em.bam" | awk '
BEGIN { OFS="\t" }
{
    if($2 == 4) next; # Пропуск невыровненных
    
    chrom = $3; start = $4 - 1; end = start + length($10);
    strand = (and($2, 16)) ? "-" : "+";
    
    prob = 0.01; mirna = "unknown";
    
    for(i=12; i<=NF; i++) {
        if($i ~ /^ZW:f:/) { split($i, a, ":"); prob = a[3]; }
        # ChiRA сохраняет имя микроРНК в QNAME 
        if(i == 12) {
            split($1, b, ":::"); 
            if(length(b) > 1) mirna = b[2];
        }
    }
    
    if(prob > 0.01) { # Отсекаем нулевой шум
        locus_id = "L1_EM_" chrom "_" start "_" end;
        print chrom, start, end, locus_id, prob, strand, mirna;
    }
}' > "${PEAKS_BED}"

# --- 5. Запуск метрик и фильтрации на Python ---
echo "[INFO] ШАГ 5: Запуск chimera_metrics.py для strand-specific фильтрации..."
"${PYTHON_BIN}" chimera_metrics.py \
    --peaks-bed "${PEAKS_BED}" \
    --active-l1-bed "${ACTIVE_L1_BED}" \
    --raw-fastq "${RAW_R1[0]}" \
    --out-parquet data/prepped/ground_truth_ago2_peaks.parquet \
    --out-audit data/lineage/chimera_em_mapping_audit.json \
    --threads "${THREADS}"

echo "[INFO] ========================================================"
echo "[INFO] Деконволюция химер и картирование успешно завершены."
echo "[INFO] ========================================================"
exit 0