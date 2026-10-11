#!/bin/bash
# scripts/00_data_ingestion_srr_only.sh
# Загрузка исключительно SRR датасетов с динамической проверкой структуры

set -euo pipefail

trap 'echo "[ERROR] Ошибка на строке $LINENO. Команда: $BASH_COMMAND" >&2; exit 1' ERR
trap 'echo "[WARNING] Прервано пользователем."; exit 1' SIGINT SIGTERM

MODE="test"
if [ $# -gt 0 ]; then
  case "$1" in
  --full) MODE="full" ;;
  --test) MODE="test" ;;
  *) echo "Использование: $0 [--full|--test]" >&2; exit 1 ;;
  esac
fi

echo "[INFO] Запуск загрузки SRR. Режим: ${MODE}"
mkdir -p data/raw/eclip

download_ena_srr() {
    local srr_id="$1"
    local target_dir="$2"

    # Формируем путь ENA (логика для 8-значных номеров, например SRR34920805)
    local prefix="${srr_id:0:6}"
    local last_digit="${srr_id: -1}"
    local base_url="https://ftp.sra.ebi.ac.uk/vol1/fastq/${prefix}/00${last_digit}/${srr_id}"

    echo "[INFO] Проверка структуры данных для ${srr_id} на сервере ENA..."

    # Используем wget --spider для проверки существования файла БЕЗ его скачивания
    if wget --spider -q "${base_url}/${srr_id}_1.fastq.gz"; then
        echo "[INFO] ${srr_id}: Обнаружены парные прочтения (paired-end). Начинаю загрузку..."
        wget -c -q --show-progress -P "${target_dir}" "${base_url}/${srr_id}_1.fastq.gz"
        wget -c -q --show-progress -P "${target_dir}" "${base_url}/${srr_id}_2.fastq.gz"
    
    elif wget --spider -q "${base_url}/${srr_id}.fastq.gz"; then
        echo "[INFO] ${srr_id}: Обнаружены одиночные прочтения (single-end). Начинаю загрузку..."
        wget -c -q --show-progress -P "${target_dir}" "${base_url}/${srr_id}.fastq.gz"
    
    else
        echo "[ERROR] ${srr_id}: Файлы не найдены по адресу ${base_url}"
        exit 1
    fi
}

if [ "${MODE}" = "full" ]; then
    echo "[INFO] Старт загрузки датасетов AGO2 chimeric eCLIP..."
    
    download_ena_srr "SRR34920805" "data/raw/eclip"
    download_ena_srr "SRR34920806" "data/raw/eclip"
    
    echo "[INFO] Все SRR успешно загружены."
else
    echo "[INFO] Режим test (холостой запуск). Для реального скачивания запустите: $0 --full"
fi

exit 0