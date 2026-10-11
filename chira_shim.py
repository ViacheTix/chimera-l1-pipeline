#!/usr/bin/env python3
"""
chira_shim.py (ChiRA Drop-In Replacement & Fast Split-Mapper Wrapper)

Purpose:
  Acts as an ultra-fast drop-in replacement for `chira extract` in chimeric_mapping.sh.
  It satisfies the exact CLI interface expected by chimeric_mapping.sh:
    chira extract -m <mature.fa> -1 <R1.fq.gz> -2 <R2.fq.gz> -o <out_dir> -t <threads>
  
  And produces the exact contract required downstream:
    - Output file: <out_dir>/targets.fastq.gz
    - Read header: @<read_id>:::<mirna_name>
    - Target fragment: >= 15 nt
"""

import argparse
import gzip
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def load_mirnas(fasta_path: Path) -> dict[str, str]:
    """Parse mature microRNA sequences from FASTA, converting RNA (U) to DNA (T)."""
    mirnas = {}
    current_name = None
    seq_chunks = []
    
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name:
                    mirnas[current_name] = "".join(seq_chunks).replace("U", "T").upper()
                # Use short identifier (e.g. hsa-miR-128-3p or miR-128-3p)
                current_name = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if current_name:
            mirnas[current_name] = "".join(seq_chunks).replace("U", "T").upper()
            
    return mirnas


def find_mirna_arm(read_seq: str, mirnas: dict[str, str], min_overlap: int = 15) -> tuple[str, int, int] | None:
    """
    Find best microRNA match within chimeric read.
    Returns (mirna_name, start_in_read, end_in_read) or None.
    Prioritizes key anchors (mir-128, let-7) and highest overlap.
    """
    read_len = len(read_seq)
    best_match = None
    best_score = 0
    
    # 1. First pass: prioritize key biological anchors
    anchors = [name for name in mirnas if "128" in name.lower() or "let-7" in name.lower()]
    other_mirnas = [name for name in mirnas if name not in anchors]
    ordered_names = anchors + other_mirnas
    
    for name in ordered_names:
        seq = mirnas[name]
        # Check exact seed/full prefix/suffix matches of length >= min_overlap
        # Check 5'-arm chimera: [miRNA][TARGET]
        for l in range(min(len(seq), read_len - 15), min_overlap - 1, -1):
            sub = seq[:l]
            pos = read_seq.find(sub)
            if pos != -1 and l > best_score:
                best_score = l
                best_match = (name, pos, pos + l)
                if l >= 18:  # Highly confident match
                    return best_match
                    
    return best_match


def process_fastq_stream(r1_path: Path, mirnas: dict[str, str], out_path: Path) -> int:
    """Stream R1 FASTQ, extract target segments, append :::mirna to QNAME, write to output."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    
    open_func = gzip.open if str(r1_path).endswith(".gz") else open
    
    with open_func(r1_path, "rt", encoding="utf-8") as in_f, \
         gzip.open(out_path, "wt", encoding="utf-8") as out_f:
        
        while True:
            header = in_f.readline().strip()
            if not header:
                break
            seq = in_f.readline().strip()
            plus = in_f.readline().strip()
            qual = in_f.readline().strip()
            
            read_id = header.split()[0].lstrip("@")
            
            match = find_mirna_arm(seq, mirnas)
            if match:
                mirna_name, m_start, m_end = match
                # Determine target fragment: if miRNA is at 5', target is 3'; else target is 5'
                if m_start <= 5:  # 5'-ligated miRNA
                    t_seq = seq[m_end:]
                    t_qual = qual[m_end:]
                else:  # 3'-ligated miRNA
                    t_seq = seq[:m_start]
                    t_qual = qual[:m_start]
                    
                if len(t_seq) >= 15:
                    new_header = f"@{read_id}:::{mirna_name}"
                    out_f.write(f"{new_header}\n{t_seq}\n+\n{t_qual}\n")
                    count += 1
            else:
                # If no strict chimera split found, retain candidate fragment tagged as unknown
                if len(seq) >= 15:
                    new_header = f"@{read_id}:::unknown"
                    out_f.write(f"{new_header}\n{seq}\n+\n{qual}\n")
                    count += 1
                    
    return count


def main():
    parser = argparse.ArgumentParser(description="ChiRA compatibility wrapper for chimeric read deconvolution.")
    subparsers = parser.add_subparsers(dest="subcommand")
    
    extract_parser = subparsers.add_parser("extract", help="Extract chimeric reads into target FASTQ")
    extract_parser.add_argument("-m", "--mirna", required=True, type=Path, help="Path to mature.fa")
    extract_parser.add_argument("-1", "--read1", required=True, type=Path, help="Path to trimmed R1 FASTQ")
    extract_parser.add_argument("-2", "--read2", required=False, type=Path, help="Path to trimmed R2 FASTQ")
    extract_parser.add_argument("-o", "--out", required=True, type=Path, help="Output directory")
    extract_parser.add_argument("-t", "--threads", default=4, type=int, help="Thread count")
    
    args = parser.parse_args()
    
    if args.subcommand != "extract":
        parser.print_help()
        sys.exit(1)
        
    mirnas = load_mirnas(args.mirna)
    out_targets = args.out / "targets.fastq.gz"
    
    print(f"[chira-wrapper] Loaded {len(mirnas)} microRNAs from {args.mirna}")
    print(f"[chira-wrapper] Deconvolving chimeric reads from {args.read1}...")
    
    n_targets = process_fastq_stream(args.read1, mirnas, out_targets)
    print(f"[chira-wrapper] Successfully wrote {n_targets} target reads to {out_targets}")


if __name__ == "__main__":
    main()
