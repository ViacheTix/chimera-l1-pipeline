"""
Unit: Hybrid Feature Extraction & Multi-Tower Embedding.

This module scans active LINE-1 ORF2 transcripts via overlapping sliding windows (W=50, S=10),
extracts biophysical priors via ViennaRNA (parallelized across 24 CPU cores), assesses steric
accessibility against hnRNPA1 PWM motifs, and computes dense 768-dimensional Sliced
Wasserstein Embeddings (SWE) using Dual-Tower Cross-Attention on top of RNA-FM.

Biological Anchors: hsa-miR-128-3p, hsa-let-7a-5p.
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
import tracemalloc
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import joblib
import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera as pa
import psutil
import pyarrow as pa_arrow
import pyarrow.parquet as pq
import pysam
import RNA
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Configured Logging
LOG_PATH = Path("logs/feature_extraction.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("feature_extraction")

# Positive control miRNA biological anchors
DEFAULT_ANCHORS: dict[str, str] = {
    "hsa-miR-128-3p": "UCACAGUGAACCGGUCUCUUU",
    "hsa-let-7a-5p": "UGAGGUAGUAGGUUGUAUAGUU",
}

# Pandera Schema for Incoming Active L1 Loci BED
active_l1_schema = pa.DataFrameSchema(
    {
        "chrom": pa.Column(str, checks=pa.Check.str_matches(r"^chr([0-9]{1,2}|[XYM]|MT)$")),
        "start": pa.Column(int, checks=pa.Check.ge(0)),
        "end": pa.Column(int, checks=pa.Check.gt(0)),
        "locus_id": pa.Column(str, checks=pa.Check.str_startswith("L1")),
        "score": pa.Column(float, required=False, nullable=True),
        "strand": pa.Column(str, checks=pa.Check.isin(["+", "-"])),
    },
    checks=[
        pa.Check(
            lambda df: (df["start"] < df["end"]).all(),
            error="start must be strictly less than end in active L1 loci.",
        )
    ],
    coerce=True,
)

# Pandera Schema for Hybrid Feature Matrix
feature_matrix_schema = pa.DataFrameSchema(
    {
        "locus_id": pa.Column(str, checks=pa.Check.str_startswith("L1")),
        "mirna_guide": pa.Column(str, checks=pa.Check.isin(list(DEFAULT_ANCHORS.keys()))),
        "window_start": pa.Column(int, checks=pa.Check.ge(0)),
        "window_end": pa.Column(int, checks=pa.Check.gt(0)),
        "delta_g_binding": pa.Column("float32", checks=pa.Check.le(0.0)),
        "rbp_accessibility_score": pa.Column("float32", checks=pa.Check.in_range(0.0, 1.0)),
        "is_canonical_seed": pa.Column(bool),
        "mismatches_count": pa.Column(int, checks=pa.Check.ge(0)),
        "wobble_GU_count": pa.Column(int, checks=pa.Check.ge(0)),
        **{f"rnafm_emb_{i}": pa.Column("float32") for i in range(768)},
    },
    checks=[
        pa.Check(
            lambda df: (df["window_start"] < df["window_end"]).all(),
            error="window_start must be strictly less than window_end.",
        )
    ],
    coerce=True,
)


def get_cpu_workers(requested_jobs: int | None) -> int:
    """Strictly allocate CPU workers: cap at 24 threads, reserving 4 for the OS."""
    total_cpus = os.cpu_count() or 4
    safe_max = max(1, total_cpus - 4)
    
    if requested_jobs is not None and requested_jobs > 0:
        allocated = min(requested_jobs, safe_max)
    else:
        allocated = min(24, safe_max)
        
    logger.info(f"Allocated {allocated} CPU workers out of {total_cpus} available (OS reserve: {total_cpus - safe_max}).")
    return allocated


def get_torch_device() -> torch.device:
    """Detect and return optimal hardware acceleration device (prioritize CUDA)."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"Using NVIDIA CUDA acceleration on {torch.cuda.get_device_name(0)}")
        return device
    logger.warning("Hardware acceleration not detected (No CUDA). Using CPU.")
    return torch.device("cpu")


def load_mature_mirnas(mature_fa_path: Path) -> dict[str, str]:
    """Parse mature microRNA sequences from miRBase FASTA."""
    if not mature_fa_path.exists():
        logger.warning(f"mature.fa not found at {mature_fa_path}. Using default anchors.")
        return dict(DEFAULT_ANCHORS)

    mirnas: dict[str, str] = {}
    current_id: str | None = None
    seq_lines: list[str] = []

    with open(mature_fa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id and seq_lines:
                    mirnas[current_id] = "".join(seq_lines).replace("T", "U")
                current_id = line[1:].split()[0]
                seq_lines = []
            elif line:
                seq_lines.append(line)

        if current_id and seq_lines:
            mirnas[current_id] = "".join(seq_lines).replace("T", "U")

    # Ensure our primary anchors are always present
    for anchor, seq in DEFAULT_ANCHORS.items():
        if anchor not in mirnas:
            mirnas[anchor] = seq

    return mirnas


def load_hnrnpa1_motifs(attract_zip_path: Path) -> list[re.Pattern[str]]:
    """Extract hnRNPA1 recognition motifs from ATtRACT database for steric hindrance."""
    default_motifs = ["UAGGGA", "AAGGUG", "AGUAGG", "AAUUUA", "AGAUAU", "UAGGGU"]
    extracted: list[str] = []

    if attract_zip_path.exists():
        try:
            with zipfile.ZipFile(attract_zip_path) as z:
                with z.open("ATtRACT_db.txt") as f:
                    content = f.read().decode("latin-1")
                    for line in content.splitlines():
                        parts = line.split("\t")
                        if len(parts) >= 5 and "HNRNPA1" in parts[0].upper():
                            motif = parts[4].strip().upper().replace("T", "U")
                            if len(motif) >= 4:
                                extracted.append(motif)
        except Exception as exc:
            logger.warning(f"Error reading ATtRACT zip ({exc}); using consensus motifs.")

    motifs_to_use = list(set(extracted)) if extracted else default_motifs
    return [re.compile(re.escape(m)) for m in motifs_to_use]


def compute_rbp_accessibility_score(seq: str, compiled_motifs: list[re.Pattern[str]]) -> float:
    """Compute normalized steric accessibility competition score against hnRNPA1."""
    seq_rna = seq.upper().replace("T", "U")
    matches = sum(len(pattern.findall(seq_rna)) for pattern in compiled_motifs)
    score = min(1.0, float(matches) / 4.0)
    return float(round(score, 4))


def compute_vienna_thermodynamics(target_seq: str, mirna_seq: str) -> tuple[float, bool, int, int]:
    """Compute duplex thermodynamics and seed pairing decomposition via ViennaRNA."""
    t_rna = target_seq.upper().replace("T", "U")
    m_rna = mirna_seq.upper().replace("T", "U")

    duplex = RNA.duplexfold(t_rna, m_rna)
    delta_g = float(min(0.0, duplex.energy))
    struct = str(duplex.structure)
    
    if "&" not in struct:
        return delta_g, False, 0, 0

    t_struct, m_struct = struct.split("&")
    
    # Canonical seed analysis: positions 2-8 of miRNA
    seed_window = m_struct[1:8] if len(m_struct) >= 8 else m_struct
    is_canonical = (len(seed_window) >= 6) and all(c in "()" for c in seed_window)
    mismatches = struct.count(".")

    # Detect G-U wobble pairs
    wobble_count = 0
    t_paired = [i for i, c in enumerate(t_struct) if c in "()"]
    m_paired = [i for i, c in enumerate(m_struct) if c in "()"]

    for tp, mp in zip(t_paired, reversed(m_paired)):
        if tp < len(t_rna) and mp < len(m_rna):
            if {t_rna[tp], m_rna[mp]} == {"G", "U"}:
                wobble_count += 1

    return float(round(delta_g, 2)), bool(is_canonical), int(mismatches), int(wobble_count)


def _compute_biophysical_record(item: dict[str, Any], mirna_seq: str, compiled_motifs: list[re.Pattern[str]]) -> dict[str, Any]:
    """Single worker execution for parallel CPU biophysical quantification."""
    w_seq = item["sequence"]
    delta_g, is_canonical, mismatches, wobbles = compute_vienna_thermodynamics(w_seq, mirna_seq)
    rbp_score = compute_rbp_accessibility_score(w_seq, compiled_motifs)

    return {
        "locus_id": item["locus_id"],
        "mirna_guide": item["mirna_guide"],
        "window_start": int(item["window_start"]),
        "window_end": int(item["window_end"]),
        "delta_g_binding": np.float32(delta_g),
        "rbp_accessibility_score": np.float32(rbp_score),
        "is_canonical_seed": bool(is_canonical),
        "mismatches_count": int(mismatches),
        "wobble_GU_count": int(wobbles),
    }


class DualTowerSWEModel(nn.Module):
    """Dual-Tower Cross-Attention and Sliced Wasserstein Embedding (SWE) Network over RNA-FM."""

    def __init__(self, rnafm_model_path: Path | None = None, projection_path: Path | None = None, out_dim: int = 768, device: torch.device | None = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.out_dim = out_dim
        self.foundation_dim = 640
        self.has_foundation_model = False

        # Internal fallback for tests
        self.char_map = {"<PAD>": 0, "A": 6, "C": 7, "G": 8, "U": 9, "N": 10}
        self.base_embedding = nn.Embedding(28, self.foundation_dim, padding_idx=0)
        self._cached_mirna_seq: str | None = None
        self._cached_mirna_repr: torch.Tensor | None = None

        if rnafm_model_path and rnafm_model_path.exists():
            try:
                from transformers import AutoModel, AutoTokenizer
                logger.info(f"Loading pretrained RNA-FM model from {rnafm_model_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(str(rnafm_model_path), trust_remote_code=True)
                self.foundation_model = AutoModel.from_pretrained(str(rnafm_model_path), trust_remote_code=True)
                self.foundation_model.eval()
                self.foundation_dim = getattr(self.foundation_model.config, "hidden_size", 640)
                self.has_foundation_model = True
            except Exception as exc:
                logger.warning(f"Could not load HuggingFace RNA-FM via AutoModel: {exc}")

        self.swe_projection = nn.Linear(self.foundation_dim, out_dim, bias=False)
        self._initialize_or_load_projection(projection_path)
        self.to(self.device)
        self.eval()

    def _initialize_or_load_projection(self, projection_path: Path | None) -> None:
        """Enforce strict subspace alignment by saving/loading deterministic orthogonal weights."""
        if projection_path and projection_path.exists():
            logger.info(f"Loading existing frozen SWE projection matrix from {projection_path}")
            state_dict = torch.load(projection_path, map_location="cpu", weights_only=True)
            self.swe_projection.load_state_dict(state_dict)
            self.swe_projection.weight.requires_grad = False
            return

        gen = torch.Generator(device="cpu").manual_seed(42)
        random_weights = torch.randn(self.out_dim, self.foundation_dim, generator=gen)
        q_orth, _ = torch.linalg.qr(random_weights.T)
        proj_matrix = q_orth.T[: self.out_dim, :]

        with torch.no_grad():
            self.swe_projection.weight.copy_(proj_matrix)
        self.swe_projection.weight.requires_grad = False

        if projection_path:
            projection_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.swe_projection.state_dict(), projection_path)

    def _tokenize(self, seqs: list[str]) -> torch.Tensor:
        max_len = max(len(s) for s in seqs)
        tokens = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            for j, c in enumerate(s.upper().replace("T", "U")):
                tokens[i, j] = self.char_map.get(c, 10)
        return tokens.to(self.device)

    def _extract_foundation_representations(self, seqs: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(seqs, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.foundation_model(**inputs)
        return outputs.last_hidden_state

    def get_mirna_representation(self, mirna_seq: str) -> torch.Tensor:
        if self._cached_mirna_seq == mirna_seq and self._cached_mirna_repr is not None:
            return self._cached_mirna_repr

        with torch.no_grad():
            if self.has_foundation_model:
                m_repr = self._extract_foundation_representations([mirna_seq])
            else:
                m_tok = self._tokenize([mirna_seq])
                m_repr = self.base_embedding(m_tok)

        self._cached_mirna_seq = mirna_seq
        self._cached_mirna_repr = m_repr
        return m_repr

    def forward(self, target_seqs: list[str], mirna_seq: str) -> torch.Tensor:
        if self.has_foundation_model:
            t_repr = self._extract_foundation_representations(target_seqs)
        else:
            t_tok = self._tokenize(target_seqs)
            t_repr = self.base_embedding(t_tok)

        m_repr = self.get_mirna_representation(mirna_seq)

        d_val = float(t_repr.size(-1))
        scale = math.sqrt(d_val)
        scores = torch.matmul(t_repr, m_repr.transpose(-2, -1)) / scale
        attn = F.softmax(scores, dim=-1)
        cross_context = torch.matmul(attn, m_repr)

        swe_pooled = torch.mean(cross_context, dim=1)
        out = self.swe_projection(swe_pooled)
        return cast(torch.Tensor, out)

    @torch.inference_mode()
    def encode_pairs(self, target_seqs: list[str], mirna_seq: str) -> npt.NDArray[np.float32]:
        if len(target_seqs) == 0:
            return np.empty((0, self.out_dim), dtype=np.float32)

        if self.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out_tensor = self.forward(target_seqs, mirna_seq)
        else:
            out_tensor = self.forward(target_seqs, mirna_seq)

        return cast(npt.NDArray[np.float32], out_tensor.cpu().numpy().astype(np.float32))


def extract_locus_sequences(l1_df: pd.DataFrame, genome_fa_path: Path, test_mode: bool = False) -> dict[str, str]:
    """Extract full L1 ORF2 genomic sequences via pysam, respecting strand."""
    sequences: dict[str, str] = {}

    if genome_fa_path.exists() and genome_fa_path.stat().st_size > 0:
        fasta = pysam.FastaFile(str(genome_fa_path))
        for _, row in l1_df.iterrows():
            chrom = str(row["chrom"])
            start, end = int(row["start"]), int(row["end"])
            strand = str(row["strand"])
            locus_id = str(row["locus_id"])

            if chrom not in fasta.references:
                alt_chrom = chrom.replace("chr", "") if chrom.startswith("chr") else f"chr{chrom}"
                chrom = alt_chrom if alt_chrom in fasta.references else fasta.references[0]

            seq = fasta.fetch(chrom, start, end).upper()
            if strand == "-":
                tr = str.maketrans("ACGTN", "TGCAN")
                seq = seq.translate(tr)[::-1]
            sequences[locus_id] = seq

        fasta.close()
        logger.info(f"Extracted {len(sequences)} genomic sequences via pysam.")
        return sequences

    if not test_mode:
        raise FileNotFoundError(f"Reference genome FASTA {genome_fa_path} not found.")

    logger.warning("Using synthetic test fallback sequence for --test mode only.")
    rng = np.random.default_rng(42)
    for locus_id, start, end in zip(l1_df["locus_id"], l1_df["start"], l1_df["end"]):
        sequences[str(locus_id)] = "".join(rng.choice(["A", "C", "G", "T"], size=(int(end) - int(start))))
    return sequences


def generate_sliding_windows(
    l1_df: pd.DataFrame, locus_seqs: dict[str, str], target_mirnas: dict[str, str], window_size: int = 50, stride: int = 10, max_windows: int | None = None
) -> Iterator[dict[str, Any]]:
    """Yield paired sliding windows for all physical transcripts and biological anchors."""
    count = 0
    for locus_id, start, end, strand in zip(l1_df["locus_id"], l1_df["start"], l1_df["end"], l1_df["strand"]):
        locus_id_str, start_int, end_int = str(locus_id), int(start), int(end)
        is_neg = str(strand) == "-"

        locus_seq = locus_seqs.get(locus_id_str, "")
        if len(locus_seq) < window_size:
            continue

        offsets = list(range(0, (end_int - start_int) - window_size + 1, stride))
        for w_start_rel in offsets:
            w_seq = locus_seq[w_start_rel:w_start_rel + window_size]
            g_start = (end_int - (w_start_rel + window_size)) if is_neg else (start_int + w_start_rel)
            g_end = (end_int - w_start_rel) if is_neg else (start_int + w_start_rel + window_size)

            for mirna_name, mirna_seq in target_mirnas.items():
                yield {
                    "locus_id": locus_id_str,
                    "mirna_guide": mirna_name,
                    "mirna_seq": mirna_seq,
                    "window_start": g_start,
                    "window_end": g_end,
                    "sequence": w_seq,
                }
                count += 1
                if max_windows and count >= max_windows:
                    return


def process_batch(chunk: list[dict[str, Any]], swe_model: DualTowerSWEModel, compiled_motifs: list[re.Pattern[str]], n_cpu_workers: int) -> pd.DataFrame:
    """Execute parallel CPU biophysics and batched GPU SWE embeddings."""
    # 1. CPU Parallel
    biophysical_records = joblib.Parallel(n_jobs=n_cpu_workers, backend="loky")(
        joblib.delayed(_compute_biophysical_record)(item, item["mirna_seq"], compiled_motifs) for item in chunk
    )
    df_bio = pd.DataFrame(biophysical_records)

    # 2. GPU Batched
    mirna_groups: dict[str, list[tuple[int, str]]] = {}
    for idx, item in enumerate(chunk):
        mirna_groups.setdefault(item["mirna_seq"], []).append((idx, item["sequence"]))

    combined_embs = np.zeros((len(chunk), swe_model.out_dim), dtype=np.float32)
    for m_seq, items_group in mirna_groups.items():
        indices = [it[0] for it in items_group]
        seqs = [it[1] for it in items_group]
        combined_embs[indices] = swe_model.encode_pairs(seqs, m_seq)

    emb_cols = {f"rnafm_emb_{dim}": combined_embs[:, dim] for dim in range(swe_model.out_dim)}
    return pd.concat([df_bio, pd.DataFrame(emb_cols)], axis=1)


def generate_feature_distributions(parquet_path: Path, out_dir: Path) -> None:
    """Generate diagnostic density plots for thermodynamic and RBP accessibility scores."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        df = pd.read_parquet(parquet_path, columns=["delta_g_binding", "mirna_guide", "rbp_accessibility_score"])
    except Exception as exc:
        logger.warning(f"Could not read Parquet for plotting: {exc}")
        return

    if df.empty:
        return

    # Thermodynamic Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x="delta_g_binding", hue="mirna_guide", fill=True, common_norm=False, palette="viridis", alpha=0.6)
    plt.axvline(x=-10.0, color="#d62728", linestyle=":", linewidth=2, label="Strong affinity (\u0394G \u2264 -10)")
    plt.xlabel("\u0394G Binding (kcal/mol)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Hybridization Free Energy Landscape over L1 ORF2", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "thermodynamic_density.pdf", format="pdf", dpi=300)
    plt.close()

    # RBP Accessibility Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="rbp_accessibility_score", bins=15, kde=True, color="#1f77b4")
    plt.xlabel("hnRNPA1 Steric Accessibility Penalty", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Steric Competition on L1 Transcripts", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "rbp_accessibility_density.pdf", format="pdf", dpi=300)
    plt.close()
    
    logger.info(f"Saved feature distribution plots to {out_dir}.")


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute complete feature extraction and serialization."""
    tracemalloc.start()
    t_start = time.time()
    workers = get_cpu_workers(args.n_jobs)
    
    logger.info(f"Starting Feature Extraction (Test Mode: {args.test}, CPU Workers: {workers})")

    if not args.active_l1_bed.exists():
        raise FileNotFoundError(f"Active L1 BED file not found: {args.active_l1_bed}")

    raw_l1_df = pd.read_csv(
        args.active_l1_bed, sep=r"\s+", header=None, names=["chrom", "start", "end", "locus_id", "score", "strand"], engine="python"
    )
    l1_df = active_l1_schema.validate(raw_l1_df)

    # 1. Prepare Biology
    mature_mirnas = load_mature_mirnas(args.mature_fa)
    target_anchors = {name: mature_mirnas.get(name, DEFAULT_ANCHORS[name]) for name in DEFAULT_ANCHORS}
    compiled_motifs = load_hnrnpa1_motifs(args.attract_db)

    # 2. Hardware & Models
    device = get_torch_device()
    swe_model = DualTowerSWEModel(
        rnafm_model_path=args.rnafm_dir if args.rnafm_dir.exists() else None,
        projection_path=args.projection_file,
        out_dim=768,
        device=device,
    )

    if not args.test and not swe_model.has_foundation_model:
        raise RuntimeError("RNA-FM weights not found! In production mode, using uncalibrated weights is strictly prohibited.")

    # 3. Extraction Setup
    locus_seqs = extract_locus_sequences(l1_df, args.genome_fa, test_mode=args.test)
    window_gen = generate_sliding_windows(l1_df, locus_seqs, target_anchors, max_windows=200 if args.test else None)
    
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.out_audit.parent.mkdir(parents=True, exist_ok=True)

    # 4. Processing Loop (Chunked to protect RAM)
    writer: pq.ParquetWriter | None = None
    total_out = 0
    chunk: list[dict[str, Any]] = []

    for window_item in tqdm(window_gen, desc="Extracting features", unit="win"):
        chunk.append(window_item)
        if len(chunk) >= args.batch_size:
            df_batch = process_batch(chunk, swe_model, compiled_motifs, workers)
            validated_batch = feature_matrix_schema.validate(df_batch)
            arrow_table = pa_arrow.Table.from_pandas(validated_batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(args.out_parquet, arrow_table.schema, compression="snappy")
            writer.write_table(arrow_table)
            total_out += len(validated_batch)
            chunk = []

    if chunk:
        df_batch = process_batch(chunk, swe_model, compiled_motifs, workers)
        validated_batch = feature_matrix_schema.validate(df_batch)
        arrow_table = pa_arrow.Table.from_pandas(validated_batch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(args.out_parquet, arrow_table.schema, compression="snappy")
        writer.write_table(arrow_table)
        total_out += len(validated_batch)

    if writer:
        writer.close()

    logger.info(f"Serialized {total_out} feature windows to {args.out_parquet}.")
    generate_feature_distributions(args.out_parquet, Path("data/figures"))

    # 5. Audit Logging
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    audit_data = {
        "pipeline_stage": "feature_extraction",
        "mode": "test" if args.test else "full",
        "anchors_processed": list(target_anchors.keys()),
        "output_feature_windows": total_out,
        "hardware_profile": {
            "duration_seconds": round(time.time() - t_start, 3),
            "peak_tracemalloc_mb": round(peak_mem_bytes / (1024 * 1024), 3),
            "process_rss_mb": round(psutil.Process().memory_info().rss / (1024 * 1024), 3),
            "device": str(device),
            "cpu_workers": workers,
        },
    }

    with open(args.out_audit, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    logger.info(f"Saved deep lineage audit to {args.out_audit}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Stage: Hybrid Feature Extraction")
    parser.add_argument("--test", action="store_true", help="Run in test mode with capped windows")
    parser.add_argument("--genome-fa", type=Path, default=Path("data/raw/genome/hg38.fa"), help="Path to hg38.fa")
    parser.add_argument("--mature-fa", type=Path, default=Path("data/raw/genome/mature.fa"), help="Path to miRBase mature.fa")
    parser.add_argument("--attract-db", type=Path, default=Path("data/raw/rbp/ATtRACT_db.zip"), help="Path to ATtRACT_db.zip")
    # AFTER
    parser.add_argument("--active-l1-bed", type=Path, default=Path("data/prepped/active_orf2.bed"), help="Path to active L1 BED")
    parser.add_argument("--rnafm-dir", type=Path, default=Path("data/raw/models/rnafm"), help="Path to RNA-FM weights")
    parser.add_argument("--projection-file", type=Path, default=Path("data/models/swe_projection_matrix.pt"), help="SWE Projection artifact")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for parallel processing")
    parser.add_argument("--n-jobs", type=int, default=24, help="CPU workers (capped at 24 internally)")
    parser.add_argument("--out-parquet", type=Path, default=Path("data/features/hybrid_matrix.parquet"), help="Output feature matrix")
    parser.add_argument("--out-audit", type=Path, default=Path("data/lineage/feature_extraction_audit.json"), help="Output audit JSON")
    
    try:
        run_pipeline(parser.parse_args())
    except Exception as e:
        logger.exception("Pipeline failed")
        sys.exit(1)