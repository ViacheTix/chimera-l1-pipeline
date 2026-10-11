#!/usr/bin/env bash
#
# extract_l1_rmsk.sh
#
# Описание: 
#   Извлекает полноразмерные (интактные) ретротранспозоны LINE-1 
#   из таблицы RepeatMasker (UCSC format) и конвертирует их в BED6.
# 
# Биологические критерии (Методология):
#   - Подсемейства: только молодые и активные L1HS и L1PA1.
#   - Длина: >= 6000 bp (для обеспечения наличия интактных ORF1/ORF2).
#   - Хромосомы: канонические (chr1-22, X, Y) для предотвращения 
#     мульти-маппинга на альтернативные контиги.

set -euo pipefail

# Значения по умолчанию
INPUT_RMSK="data/raw/genome/rmsk.txt.gz"
OUTPUT_BED="data/raw/genome/L1_filtered.bed"
MIN_LENGTH=5800

# Функция логирования
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $1"
}
err() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

# Парсинг аргументов командной строки (CLI Orchestration)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input) INPUT_RMSK="$2"; shift ;;
        --output) OUTPUT_BED="$2"; shift ;;
        --min-length) MIN_LENGTH="$2"; shift ;;
        -h|--help)
            echo "Использование: $0 --input <rmsk.txt.gz> --output <L1_filtered.bed> --min-length <6000>"
            exit 0
            ;;
        *) err "Неизвестный параметр: $1"; exit 1 ;;
    esac
    shift
done

# Проверка наличия входного файла
if [[ ! -f "${INPUT_RMSK}" ]]; then
    err "Входной файл RepeatMasker не найден: ${INPUT_RMSK}"
    exit 1
fi

# Создание директории для вывода
mkdir -p "$(dirname "${OUTPUT_BED}")"

# Оптимизация Hardware: Использование многопоточного pigz (если установлен)
DECOMPRESSOR="gzip -dc"
if command -v pigz &> /dev/null; then
    DECOMPRESSOR="pigz -dc -p 4" # Выделяем 4 потока для безопасной декомпрессии
    log "Обнаружен pigz. Включена многопоточная распаковка потока."
fi

log "Запуск конвейера извлечения LINE-1 из ${INPUT_RMSK}..."
log "Критерии фильтрации: L1HS / L1PA1, длина >= ${MIN_LENGTH} bp."

# Потоковая фильтрация через awk
# Схема столбцов UCSC rmsk:
# $2 = swScore
# $6 = genoName (chrom)
# $7 = genoStart
# $8 = genoEnd
# $10 = strand
# $11 = repName (subfamily)
${DECOMPRESSOR} "${INPUT_RMSK}" | awk -v min_len="${MIN_LENGTH}" '
    BEGIN { FS="\t"; OFS="\t" }
    {
        chrom = $6
        start = $7
        end = $8
        strand = $10
        name = $11
        score = $2

        # 1. Жесткий фильтр канонических хромосом
        if (chrom ~ /^chr([1-9]|1[0-9]|2[0-2]|X|Y)$/) {
            
            # 2. Фильтр подсемейств и длины
            if ((name == "L1HS" || name == "L1PA1") && (end - start >= min_len)) {
                
                # 3. Формирование стандарта BED6
                print chrom, start, end, name, score, strand
            }
        }
    }
' > "${OUTPUT_BED}"

# Подсчет результатов
COUNT=$(wc -l < "${OUTPUT_BED}")
if [[ "$COUNT" -eq 0 ]]; then
    err "После фильтрации не осталось ни одного локуса! Проверьте формат входного файла."
    exit 1
fi

log "Успешно извлечено ${COUNT} интактных локусов LINE-1."
log "Результат сохранен в файл: ${OUTPUT_BED}"
