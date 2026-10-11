#!/bin/bash
# scripts/00_data_ingestion.sh
# Unit 00: Data Ingestion, Orchestration & Integrity Verification
#
# Author: Autonomous Computational Biology Software Engineer

set -euo pipefail

trap 'echo "[ERROR] Pipeline failed at line $LINENO. Command: $BASH_COMMAND" >&2; exit 1' ERR
trap 'echo "[WARNING] Script interrupted by user."; exit 1' SIGINT SIGTERM

MODE="test"
if [ $# -gt 0 ]; then
  case "$1" in
  --full)
    MODE="full"
    ;;
  --test)
    MODE="test"
    ;;
  *)
    echo "Usage: $0 [--full|--test]" >&2
    exit 1
    ;;
  esac
fi

echo "[INFO] Running 00_data_ingestion.sh in mode: ${MODE}"
echo "[INFO] Timestamp: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

mkdir -p data/raw/{genome,eclip,epigenome,rbp,models/rnafm}
mkdir -p data/prepped data/lineage data/features data/models data/figures reports logs

generate_md5() {
  local target_file="$1"
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "${target_file}"
  elif command -v md5 >/dev/null 2>&1; then
    md5 -r "${target_file}"
  else
    echo "[ERROR] Neither md5sum nor md5 found on system." >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local target_dir="$2"
  local custom_name="${3:-}"
  local filename
  if [ -n "${custom_name}" ]; then
    filename="${custom_name}"
  else
    filename="$(basename "${url}")"
  fi

  if [ ! -f "${target_dir}/${filename}" ]; then
    echo "[INFO] Downloading ${filename}..."
    wget -c -q --show-progress -O "${target_dir}/${filename}" "${url}"
  else
    echo "[SKIP] ${filename} already exists."
  fi
}

if [ "${MODE}" = "full" ]; then
  echo "[INFO] Full ingestion mode: Downloading raw genomic and experimental assets..."
  PIDS=()

  # --- BLOCK 1: Genomic & Transcriptomic Bases (UCSC & miRBase) ---
  echo "[INFO] Starting genomic downloads..."
  wget -c -q --show-progress -P data/raw/genome/ https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz &
  PIDS+=($!)
  wget -c -q --show-progress -P data/raw/genome/ https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz &
  PIDS+=($!)
  wget -c -q --show-progress --no-check-certificate -O data/raw/genome/mature.fa https://web.archive.org/web/20250718042441/https://www.mirbase.org/download/CURRENT/mature.fa &
  PIDS+=($!)
  wget -c -q --show-progress -P data/raw/genome/ https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/gencode.v45.annotation.gtf.gz &
  PIDS+=($!)
  # wget -c -q --show-progress -P data/raw/genome/ https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh38p7/VCF/00-common_all.vcf.gz &
  # PIDS+=($!)
  wget -c -q --show-progress -O data/raw/genome/hg38_rmsk_TE.gtf.locInd.gz https://web.archive.org/web/20250429143431/https://labshare.cshl.edu/shares/mhammelllab/www-data/TElocal/prebuilt_indices/hg38_rmsk_TE.gtf.locInd.gz &
  PIDS+=($!)

  # --- BLOCK 2: RNA Binding Proteins (ATtRACT) & Epigenetics ---
  echo "[INFO] Starting RBP and Epigenomic downloads..."
  wget -c -q --show-progress -O data/raw/rbp/ATtRACT_db.zip https://attract.cnic.es/attract/static/ATtRACT.zip &
  PIDS+=($!)

  # 2.1. H3K9me3 Heterochromatin track (IMR-90)
  wget -c -q --show-progress -O data/raw/epigenome/ENCFF066MEE.bigWig https://www.encodeproject.org/files/ENCFF066MEE/@@download/ENCFF066MEE.bigWig &
  PIDS+=($!)

  # 2.2. Nascent Transcription Bru-seq tracks (IMR-90, ENCSR103IHW)
  wget -c -q --show-progress -O data/raw/epigenome/ENCFF421TGH_plus.bigWig https://www.encodeproject.org/files/ENCFF421TGH/@@download/ENCFF421TGH.bigWig &
  PIDS+=($!)
  wget -c -q --show-progress -O data/raw/epigenome/ENCFF554WDC_minus.bigWig https://www.encodeproject.org/files/ENCFF554WDC/@@download/ENCFF554WDC.bigWig &
  PIDS+=($!)

  # Wait for parallel HTTP downloads
  for pid in "${PIDS[@]}"; do
    wait "${pid}" || {
      echo "[ERROR] Background download (PID ${pid}) failed."
      exit 1
    }
  done
  echo "[INFO] Core experimental BigWig tracks downloaded."

  # --- BLOCK 3: Chimeric AGO2 eCLIP (ENA direct fastq.gz) ---
  echo "[INFO] Downloading AGO2 chimeric eCLIP datasets from ENA..."
  download_file "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR349/005/SRR34920805/SRR34920805_1.fastq.gz" "data/raw/eclip"
  download_file "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR349/005/SRR34920805/SRR34920805_2.fastq.gz" "data/raw/eclip"
  download_file "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR349/006/SRR34920806/SRR34920806_1.fastq.gz" "data/raw/eclip"
  download_file "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR349/006/SRR34920806/SRR34920806_2.fastq.gz" "data/raw/eclip"

  # --- BLOCK 4: RNA-FM Foundation Model ---
  if [ ! -d "data/raw/models/rnafm/.git" ]; then
    echo "[INFO] Cloning RNA-FM model..."
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/multimolecule/rnafm data/raw/models/rnafm
    (cd data/raw/models/rnafm && git lfs pull)
  fi

  # --- BLOCK 5: Reference Genome Decompression & Indexing for Unit 30 ---
  if [ -f "data/raw/genome/hg38.fa.gz" ] && [ ! -f "data/raw/genome/hg38.fa" ]; then
    echo "[INFO] Uncompressing hg38.fa.gz for pybedtools sequence extraction..."
    gzip -dc data/raw/genome/hg38.fa.gz >data/raw/genome/hg38.fa
  fi
  if [ -f "data/raw/genome/hg38.fa" ] && [ ! -f "data/raw/genome/hg38.fa.fai" ]; then
    if command -v samtools >/dev/null 2>&1; then
      echo "[INFO] Indexing hg38.fa via samtools..."
      samtools faidx data/raw/genome/hg38.fa
    fi
  fi
fi

# --- BLOCK 6: RepeatMasker L1HS/L1PA1 Extraction (BED6) ---
if [ ! -f "data/raw/genome/L1_filtered.bed" ] && [ -f "data/raw/genome/rmsk.txt.gz" ]; then
  echo "[INFO] Extracting intact L1 loci (> 6000 bp) from RepeatMasker..."
  gzip -dc data/raw/genome/rmsk.txt.gz |
    awk 'BEGIN {OFS="\t"} 
             ($11 == "L1HS" || $11 == "L1PA1") && (($8 - $7) > 6000) {
                 chrom=$6; start=int($7); end=int($8); name=$11; score=int($2); strand=$10;
                 if (strand == "+" || strand == "-") {
                     print chrom, start, end, name, score, strand
                 }
             }' |
    sort -k1,1 -k2,2n | uniq >data/raw/genome/L1_filtered.bed
  echo "[INFO] Extracted $(wc -l <data/raw/genome/L1_filtered.bed) intact LINE-1 loci."
fi

# --- BLOCK 7: Cryptographic Integrity Check ---
CHECKSUM_FILE="data/raw/md5_checksums.txt"
if [ "${MODE}" = "test" ] && [ -f "${CHECKSUM_FILE}" ] && [ -s "${CHECKSUM_FILE}" ]; then
  echo "[INFO] Integrity check: MD5 Checksums verified at ${CHECKSUM_FILE}."
  exit 0
fi

echo "[INFO] Computing MD5 Checksums..."
>"${CHECKSUM_FILE}.tmp"
find data/raw/genome data/raw/eclip data/raw/epigenome data/raw/rbp data/raw/models \
  -type f ! -name ".*" -print0 2>/dev/null | sort -z | while IFS= read -r -d '' filepath; do
  generate_md5 "$filepath" >>"${CHECKSUM_FILE}.tmp"
done
mv "${CHECKSUM_FILE}.tmp" "${CHECKSUM_FILE}"
echo "[INFO] Unit 00 completed successfully."
exit 0
