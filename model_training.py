#!/usr/bin/env python3
"""
Pipeline Stage: XGBoost Model Training & Statistical Validation
Methodology: Binary classification of in vivo AGO2-LINE-1 binding using RNA-FM and thermodynamic features.

Fixes & Upgrades:
- Strict 1:1 balancing for training, but extracting a massive N=10^4 background set for the Null Distribution.
- F1-Score calculation at the optimal probability threshold.
- Empirical p-value calculation via Monte Carlo surrogate.
- Benjamini-Hochberg (FDR) correction for multiple testing (q-value <= 0.05).
- Export of statistically significant binding sites for downstream biological analysis.
"""

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import xgboost as xgb
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# Ограничения аппаратной среды
MAX_CPU_THREADS = 24
RANDOM_STATE = 42
BACKGROUND_N = 10000  # Размер генеральной совокупности для Монте-Карло

# Настройка логирования
LOG_PATH = Path("logs/model_training.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("model_training")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_xgboost_device() -> Tuple[str, str]:
    """Определяет наличие GPU RTX 3070 для аппаратного ускорения XGBoost."""
    if torch.cuda.is_available():
        try:
            clf = xgb.XGBClassifier(n_estimators=1, max_depth=1, tree_method="hist", device="cuda")
            clf.fit(np.zeros((2, 2)), np.array([0, 1]))
            logger.info("Hardware acceleration verified: XGBoost using CUDA (RTX 3070).")
            return "hist", "cuda"
        except Exception as exc:
            logger.warning(f"CUDA validation failed: {exc}. Falling back to CPU.")
    logger.info(f"Using CPU mode with strict {MAX_CPU_THREADS} thread limit.")
    return "hist", "cpu"


def prepare_datasets(
    features_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    exclusion_buffer_bp: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разделяет данные на:
    1. balanced_df: Строгий 1:1 баланс для XGBoost (предотвращение class bias).
    2. background_df: Массивный пул (N=10^4) истинно отрицательных окон для Монте-Карло.
    3. positive_df: Все истинные сайты AGO2 для финального FDR-скрининга.
    """
    logger.info("Splitting datasets and applying Flank Buffer Exclusion...")
    
    # Хеширование пиков для быстрого поиска
    peaks_by_target = {}
    for _, row in peaks_df.iterrows():
        key = (str(row["locus_id"]), str(row["mirna_guide"]))
        peaks_by_target.setdefault(key, []).append((int(row["peak_start"]), int(row["peak_end"])))

    train_indices = []
    y_labels = []
    background_indices = []
    positive_indices = []
    
    rng = np.random.default_rng(RANDOM_STATE)
    unique_mirnas = features_df["mirna_guide"].unique()
    
    for mirna in unique_mirnas:
        idx_mask = features_df["mirna_guide"] == mirna
        sub_indices = features_df.index[idx_mask].to_numpy()
        
        pos_idx, neg_idx = [], []
        
        for idx in tqdm(sub_indices, desc=f"Mapping overlaps: {mirna}", leave=False):
            row = features_df.iloc[idx]
            key = (str(row["locus_id"]), mirna)
            target_peaks = peaks_by_target.get(key, [])
            
            w_s, w_e = int(row["window_start"]), int(row["window_end"])
            
            # Позитивный класс: физическое перекрытие
            if any(w_s < p_e and w_e > p_s for p_s, p_e in target_peaks):
                pos_idx.append(idx)
                continue
                
            # Исключение окон на границах пиков (Exclusion Buffer Zone)
            is_near_flank = any(
                (w_e > (p_s - exclusion_buffer_bp) and w_s < p_s) or 
                (w_s < (p_e + exclusion_buffer_bp) and w_e > p_e)
                for p_s, p_e in target_peaks
            )
            if not is_near_flank:
                neg_idx.append(idx)
                
        if not pos_idx or not neg_idx:
            logger.warning(f"Skipping {mirna}: Missing valid windows.")
            continue
            
        positive_indices.extend(pos_idx)
            
        # 1. Формирование сбалансированной обучающей выборки (1:1)
        n_samples = min(len(pos_idx), len(neg_idx))
        sampled_pos = rng.choice(pos_idx, size=n_samples, replace=False)
        sampled_neg = rng.choice(neg_idx, size=n_samples, replace=False)
        
        train_indices.extend(sampled_pos)
        y_labels.extend([1] * n_samples)
        train_indices.extend(sampled_neg)
        y_labels.extend([0] * n_samples)
        
        # 2. Формирование пула для нулевого распределения (Остаток негативов)
        remaining_neg = list(set(neg_idx) - set(sampled_neg))
        n_bg = min(BACKGROUND_N, len(remaining_neg))
        if n_bg > 0:
            bg_sampled = rng.choice(remaining_neg, size=n_bg, replace=False)
            background_indices.extend(bg_sampled)
            
        logger.info(f"{mirna}: Train={n_samples*2} (1:1), Background={n_bg}, Total Positives={len(pos_idx)}")

    # Перемешивание обучающей выборки
    perm = rng.permutation(len(train_indices))
    train_indices = np.array(train_indices)[perm]
    y_labels = np.array(y_labels, dtype=np.int32)[perm]
    
    balanced_df = features_df.iloc[train_indices].copy().reset_index(drop=True)
    balanced_df['target_label'] = y_labels
    
    background_df = features_df.iloc[background_indices].copy().reset_index(drop=True)
    positive_df = features_df.iloc[positive_indices].copy().reset_index(drop=True)
    
    return balanced_df, background_df, positive_df


def extract_numeric_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Векторизация DataFrame в float32 матрицу для экономии RAM."""
    exclude_cols = {"locus_id", "mirna_guide", "window_start", "window_end", "chrom", "strand", "target_label"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    sub_df = df[feature_cols].copy()
    for col in sub_df.select_dtypes(include="bool").columns:
        sub_df[col] = sub_df[col].astype(np.float32)
        
    return sub_df.to_numpy(dtype=np.float32), feature_cols


def optimize_hyperparameters(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
    tree_method: str, device: str, n_trials: int
) -> Dict[str, Any]:
    """Bayesian Optimization (Optuna) с использованием StratifiedGroupKFold."""
    logger.info(f"Initiating Optuna TPE Hyperparameter search ({n_trials} trials)...")
    
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X, y, groups=groups))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),  # L1
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 20.0, log=True), # L2
            "tree_method": tree_method,
            "device": device,
            "n_jobs": MAX_CPU_THREADS if device == "cpu" else 1,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": RANDOM_STATE
        }

        pr_aucs = []
        for train_idx, val_idx in splits:
            model = xgb.XGBClassifier(n_estimators=100, **params)
            model.fit(X[train_idx], y[train_idx], verbose=False)
            preds = model.predict_proba(X[val_idx])[:, 1]
            
            precision, recall, _ = precision_recall_curve(y[val_idx], preds)
            pr_aucs.append(auc(recall, precision))
            
        return float(np.mean(pr_aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    
    with tqdm(total=n_trials, desc="Optuna Tuning") as pbar:
        def callback(study, trial):
            pbar.set_postfix(Best_PR_AUC=f"{study.best_value:.4f}")
            pbar.update(1)
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
        
    logger.info(f"Optimal Hyperparameters found. Best PR-AUC: {study.best_value:.4f}")
    return study.best_trial.params


def statistical_validation_and_fdr(
    final_model: xgb.XGBClassifier,
    positive_df: pd.DataFrame,
    background_df: pd.DataFrame,
    out_plots_dir: Path,
    out_table_path: Path
) -> None:
    """
    Биоинформатическая валидация:
    1. Скоринг пула Монте-Карло (создание нулевого распределения).
    2. Скоринг экспериментально подтвержденных локусов (positives).
    3. Расчет эмпирического p-value для каждого предсказания.
    4. Benjamini-Hochberg FDR коррекция (q-value).
    """
    logger.info("Executing Monte Carlo Surrogate & FDR Correction...")
    
    X_pos, _ = extract_numeric_matrix(positive_df)
    X_bg, _ = extract_numeric_matrix(background_df)
    
    # Предсказания вероятности
    positive_df = positive_df.copy()
    background_df = background_df.copy()
    
    positive_df["xgb_score"] = final_model.predict_proba(X_pos)[:, 1]
    background_df["xgb_score"] = final_model.predict_proba(X_bg)[:, 1]
    
    unique_mirnas = positive_df["mirna_guide"].unique()
    sns.set_theme(style="whitegrid")
    
    significant_sites_list = []
    
    for mirna in unique_mirnas:
        pos_scores = positive_df[positive_df["mirna_guide"] == mirna]["xgb_score"].to_numpy()
        bg_scores = background_df[background_df["mirna_guide"] == mirna]["xgb_score"].to_numpy()
        
        if len(bg_scores) < 100:
            logger.warning(f"Not enough background samples for {mirna} to compute p-values.")
            continue
            
        # Векторизованный расчет эмпирического p-value
        # Доля случайных последовательностей, чей скор превышает скор кандидата
        p_values = np.array([np.sum(bg_scores >= s) / len(bg_scores) for s in pos_scores])
        p_values = np.clip(p_values, 1e-6, 1.0) # Защита от p=0
        
        # Поправка на множественное тестирование Бенджамини-Хохберга (FDR)
        reject, q_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        
        # Сохранение результатов
        mirna_pos_df = positive_df[positive_df["mirna_guide"] == mirna].copy()
        mirna_pos_df["p_value"] = p_values
        mirna_pos_df["q_value"] = q_values
        mirna_pos_df["is_significant"] = reject
        
        significant_sites = mirna_pos_df[mirna_pos_df["is_significant"]].copy()
        significant_sites_list.append(significant_sites)
        
        sig_count = len(significant_sites)
        logger.info(f"{mirna}: Found {sig_count}/{len(pos_scores)} significant sites (FDR q <= 0.05).")
        
        # Визуализация нулевого распределения
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(bg_scores, fill=True, color="#1f77b4", label=f"Null Distribution (N={len(bg_scores)})", ax=ax)
        sns.kdeplot(pos_scores, fill=True, color="#ff7f0e", label=f"True {mirna} eCLIP Sites", ax=ax)
        
        median_pos = np.median(pos_scores)
        ax.axvline(median_pos, color="red", linestyle="--", label=f"Median Positive Score")
        ax.set(xlabel="XGBoost Binding Probability", ylabel="Density", 
               title=f"Statistical Binding Validation (Monte Carlo Surrogate) - {mirna}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_plots_dir / f"null_distribution_{mirna}.pdf", dpi=300)
        plt.close(fig)

    # Экспорт достоверных биологических сайтов
    if significant_sites_list:
        final_sig_df = pd.concat(significant_sites_list, ignore_index=True)
        export_cols = ["locus_id", "chrom", "window_start", "window_end", "strand", 
                       "mirna_guide", "xgb_score", "p_value", "q_value"]
        
        out_table_path.parent.mkdir(parents=True, exist_ok=True)
        final_sig_df[export_cols].to_csv(out_table_path, sep="\t", index=False)
        logger.info(f"Successfully exported {len(final_sig_df)} FDR-significant sites to {out_table_path}")
    else:
        logger.warning("No statistically significant sites passed FDR correction.")


def evaluate_cv_metrics(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
    mirnas: np.ndarray, best_params: Dict[str, Any], 
    tree_method: str, device: str, out_plots_dir: Path
):
    """CV Оценка: PR-AUC, ROC-AUC и расчет F1-score для каждой микроРНК."""
    out_plots_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y), dtype=np.float32)
    
    logger.info("Evaluating model metrics (PR-AUC, ROC-AUC, Max F1-score)...")
    for train_idx, val_idx in tqdm(cv.split(X, y, groups=groups), total=5, desc="CV Folds"):
        model = xgb.XGBClassifier(
            n_estimators=150, tree_method=tree_method, device=device,
            n_jobs=MAX_CPU_THREADS if device == "cpu" else 1,
            objective="binary:logistic", random_state=RANDOM_STATE, **best_params
        )
        model.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        
    for mirna in np.unique(mirnas):
        mask = mirnas == mirna
        y_true_sub, y_pred_sub = y[mask], oof_preds[mask]
        
        if len(y_true_sub) < 10: continue
            
        precision, recall, thresholds = precision_recall_curve(y_true_sub, y_pred_sub)
        fpr, tpr, _ = roc_curve(y_true_sub, y_pred_sub)
        
        pr_auc = auc(recall, precision)
        roc_auc = auc(fpr, tpr)
        
        # Расчет оптимального F1-score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        max_f1 = np.max(f1_scores)
        best_thresh = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
        
        logger.info(f"[{mirna}] PR-AUC: {pr_auc:.3f} | ROC-AUC: {roc_auc:.3f} | Best F1: {max_f1:.3f} (thresh: {best_thresh:.3f})")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, color="#d62728", lw=2, label=f"PR Curve (AUC = {pr_auc:.3f})")
        ax.axhline(0.5, linestyle="--", color="gray", label="Baseline")
        ax.set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall Curve: {mirna}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_plots_dir / f"pr_curve_{mirna}.pdf", dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Model Training, CV Metrics and FDR Validation for LINE-1/miRNA interactions.")
    parser.add_argument("--features", type=Path, required=True, help="Input: Extracted features parquet (hybrid_matrix.parquet)")
    parser.add_argument("--peaks", type=Path, required=True, help="Input: AGO2 Ground Truth peaks parquet")
    parser.add_argument("--out-model", type=Path, default=Path("data/models/xgb_mirna_l1.json"), help="Output model JSON path")
    parser.add_argument("--out-plots", type=Path, default=Path("data/figures/"), help="Directory for metric plots")
    parser.add_argument("--out-sites", type=Path, default=Path("data/results/significant_ago2_sites.tsv"), help="Output for FDR validated sites")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    args = parser.parse_args()

    logger.info(f"Loading data into RAM (Limit 32GB)...")
    features_df = pd.read_parquet(args.features)
    peaks_df = pd.read_parquet(args.peaks)
    
    # 1. Предобработка и контроль Data Leakage
    balanced_df, background_df, positive_df = prepare_datasets(features_df, peaks_df)
    
    # Очистка RAM от исходного тяжелого DataFrame
    del features_df
    gc.collect() 
    
    X_train, feature_names = extract_numeric_matrix(balanced_df)
    y_train = balanced_df["target_label"].to_numpy()
    groups = balanced_df["locus_id"].to_numpy()
    mirnas = balanced_df["mirna_guide"].to_numpy()
    
    # 2. Детекция железа
    tree_method, device = get_xgboost_device()
    
    # 3. Баесовская оптимизация параметров
    best_params = optimize_hyperparameters(X_train, y_train, groups, tree_method, device, args.n_trials)
    
    # 4. Расчет классических ML метрик (Cross-Validation)
    evaluate_cv_metrics(X_train, y_train, groups, mirnas, best_params, tree_method, device, args.out_plots)
    
    # 5. Тренировка финальной продакшен-модели
    logger.info("Training final production XGBoost model on 100% of balanced data...")
    final_model = xgb.XGBClassifier(
        n_estimators=150, tree_method=tree_method, device=device,
        n_jobs=MAX_CPU_THREADS if device == "cpu" else 1,
        objective="binary:logistic", random_state=RANDOM_STATE, **best_params
    )
    final_model.fit(X_train, y_train)
    
    # Сохранение весов и метаданных
    booster = final_model.get_booster()
    booster.feature_names = feature_names
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(args.out_model)
    logger.info(f"Model saved to {args.out_model}")
    
    # Очистка памяти перед инференсом
    del X_train, balanced_df
    gc.collect()
    
    # 6. Строгая биологическая валидация (FDR, q-value)
    statistical_validation_and_fdr(final_model, positive_df, background_df, args.out_plots, args.out_sites)
    
    logger.info("Pipeline unit 100% completed successfully.")

if __name__ == "__main__":
    main()