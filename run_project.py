#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_project.py
==============
Full pipeline runner for the Amazon Product Recommendation System.

Executes every stage of the ML workflow end-to-end:
  1. Data loading and exploration
  2. Train / test split
  3. Surprise dataset initialisation
  4. Training five recommendation models
  5. Model comparison and selection
  6. Sample recommendations
  7. Saving charts and the best model

Run from the project root:
    python run_project.py
"""

import os
import sys
import pickle
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- safe for all environments
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Ensure src/ is importable regardless of working directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split
from surprise import Reader, Dataset, KNNBasic, SVD

from src.utils import smart_display
from src.rank_recommender import RankRecommendationSystem
from src.cf_recommender import CFRecommendationSystem
from src.hybrid_recommender import HybridRecommendationSystem
from src.model_eval_functions import evaluate_model


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def section(title: str, step: int = None) -> None:
    """Print a bold section header."""
    label = f"[STEP {step}] " if step else ""
    bar = "=" * 80
    print(f"\n{bar}")
    print(f"  {label}{title}")
    print(bar)


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def info(msg: str) -> None:
    print(f"      {msg}")


def save_charts(df: pd.DataFrame, figures_dir: str) -> None:
    """Generate and save EDA charts to the figures directory."""
    os.makedirs(figures_dir, exist_ok=True)

    palette = sns.color_palette("Set2")
    rating_order = [1.0, 2.0, 3.0, 4.0, 5.0]

    # --- Chart 1: Rating Distribution ---
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = df["rating"].value_counts().reindex(rating_order, fill_value=0)
    bars = ax.bar(
        [str(int(r)) for r in rating_order],
        counts.values,
        color=palette[:5],
        edgecolor="white",
        linewidth=1.5,
    )
    total = len(df)
    for bar, cnt in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{cnt / total * 100:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_title("Rating Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Star Rating", fontsize=13)
    ax.set_ylabel("Number of Reviews", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    chart1_path = os.path.join(figures_dir, "rating_distribution.png")
    plt.savefig(chart1_path, dpi=150, bbox_inches="tight")
    plt.close()
    ok(f"Saved chart: {os.path.basename(chart1_path)}")

    # --- Chart 2: Reviews per User (log scale) ---
    reviews_per_user = df.groupby("user_id")["rating"].count()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(reviews_per_user, bins=40, kde=True, color=palette[1],
                 edgecolor="white", ax=ax)
    ax.set_yscale("log")
    ax.axvline(reviews_per_user.mean(), color="red", linestyle="--",
               linewidth=1.5, label=f"Mean: {reviews_per_user.mean():.1f}")
    ax.axvline(reviews_per_user.median(), color="blue", linestyle="-",
               linewidth=1.5, label=f"Median: {reviews_per_user.median():.1f}")
    ax.set_title("Reviews per User (log scale)", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Reviews", fontsize=13)
    ax.set_ylabel("Users (log)", fontsize=13)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    chart2_path = os.path.join(figures_dir, "reviews_per_user.png")
    plt.savefig(chart2_path, dpi=150, bbox_inches="tight")
    plt.close()
    ok(f"Saved chart: {os.path.basename(chart2_path)}")

    # --- Chart 3: Reviews per Product ---
    reviews_per_prod = df.groupby("prod_id")["rating"].count()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(reviews_per_prod, bins=40, kde=True, color=palette[2],
                 edgecolor="white", ax=ax)
    ax.set_yscale("log")
    ax.axvline(reviews_per_prod.mean(), color="red", linestyle="--",
               linewidth=1.5, label=f"Mean: {reviews_per_prod.mean():.1f}")
    ax.axvline(reviews_per_prod.median(), color="blue", linestyle="-",
               linewidth=1.5, label=f"Median: {reviews_per_prod.median():.1f}")
    ax.set_title("Reviews per Product (log scale)", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Reviews", fontsize=13)
    ax.set_ylabel("Products (log)", fontsize=13)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    chart3_path = os.path.join(figures_dir, "reviews_per_product.png")
    plt.savefig(chart3_path, dpi=150, bbox_inches="tight")
    plt.close()
    ok(f"Saved chart: {os.path.basename(chart3_path)}")


# ---------------------------------------------------------------------------
# STEP 1 -- LOAD DATA
# ---------------------------------------------------------------------------
section("Loading & Exploring Data", step=1)

data_path = os.path.join(PROJECT_ROOT, "data", "processed", "processed_data.pkl")
with open(data_path, "rb") as f:
    df = pickle.load(f)

n_users = df["user_id"].nunique()
n_products = df["prod_id"].nunique()
n_interactions = len(df)
sparsity = (1 - n_interactions / (n_users * n_products)) * 100

ok("Dataset loaded successfully!")
info(f"Shape           : {df.shape}")
info(f"Unique users    : {n_users:,}")
info(f"Unique products : {n_products:,}")
info(f"Total ratings   : {n_interactions:,}")
info(f"Sparsity        : {sparsity:.2f}%")
info(f"Rating range    : {df['rating'].min():.0f} - {df['rating'].max():.0f} stars")
info(f"Average rating  : {df['rating'].mean():.2f}")

print("\n  Sample rows:")
print(df.head(5).to_string(index=False))

# ---------------------------------------------------------------------------
# STEP 2 -- EDA CHARTS
# ---------------------------------------------------------------------------
section("Generating EDA Charts", step=2)

figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
save_charts(df, figures_dir)

# ---------------------------------------------------------------------------
# STEP 3 -- TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
section("Train / Test Split", step=3)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
ok(f"Train set : {len(train_df):,} interactions")
ok(f"Test  set : {len(test_df):,} interactions")

# ---------------------------------------------------------------------------
# STEP 4 -- SURPRISE DATASET INIT
# ---------------------------------------------------------------------------
section("Initialising Surprise Framework", step=4)

reader = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
dataset = Dataset.load_from_df(df[["user_id", "prod_id", "rating"]], reader)
train_dataset = Dataset.load_from_df(train_df[["user_id", "prod_id", "rating"]], reader)
test_dataset = Dataset.load_from_df(test_df[["user_id", "prod_id", "rating"]], reader)

trainset = train_dataset.build_full_trainset()
testset = test_dataset.build_full_trainset().build_testset()

ok(f"Rating scale : {reader.rating_scale}")
ok(f"Trainset     : {trainset.n_ratings:,} ratings")
ok(f"Testset      : {len(testset):,} ratings")

# ---------------------------------------------------------------------------
# STEP 5 -- TRAIN MODELS
# ---------------------------------------------------------------------------
section("Training Recommendation Models", step=5)

model_results: dict = {}
best_rmse = float("inf")
best_model_name = None
k = 10  # top-K for ranking metrics

# -- Model 1: Rank-Based ----------------------------------------------------
print("\n  -- [1/5] Rank-Based (Bayesian Average) --")
try:
    t0 = time.time()
    model_rank = RankRecommendationSystem(method="bayesian")
    model_rank.compute_scores(train_df, gb_feature="prod_id", filter_feature="rating")
    train_met, test_met = model_rank.evaluate(trainset, testset, k=k, th=3.5)
    model_results["Rank-Based"] = test_met
    rmse_val = test_met.get("RMSE", None)
    if rmse_val and rmse_val < best_rmse:
        best_rmse, best_model_name = rmse_val, "Rank-Based"
    ok(f"Rank-Based trained in {time.time()-t0:.1f}s | Test RMSE: {rmse_val:.4f}")
except Exception as e:
    print(f"  [ERROR] Rank-Based failed: {e}")

# -- Model 2: User-User CF --------------------------------------------------
print("\n  -- [2/5] User-User Collaborative Filtering --")
try:
    t0 = time.time()
    model_cf_uu = CFRecommendationSystem(
        train_df, algo_class=KNNBasic,
        params_grid={"k": 40, "sim_options": {"name": "cosine", "user_based": True}},
    )
    model_cf_uu.fit_knn(trainset)
    train_met_uu, test_met_uu = model_cf_uu.evaluate(trainset, testset, k=k, th=3.5)
    model_results["User-User CF"] = test_met_uu
    rmse_val = test_met_uu.get("RMSE", None)
    if rmse_val and rmse_val < best_rmse:
        best_rmse, best_model_name = rmse_val, "User-User CF"
    ok(f"User-User CF trained in {time.time()-t0:.1f}s | Test RMSE: {rmse_val:.4f}")
except Exception as e:
    print(f"  [ERROR] User-User CF failed: {e}")

# -- Model 3: Item-Item CF --------------------------------------------------
print("\n  -- [3/5] Item-Item Collaborative Filtering --")
try:
    t0 = time.time()
    model_cf_ii = CFRecommendationSystem(
        train_df, algo_class=KNNBasic,
        params_grid={"k": 40, "sim_options": {"name": "cosine", "user_based": False}},
    )
    model_cf_ii.fit_knn(trainset)
    train_met_ii, test_met_ii = model_cf_ii.evaluate(trainset, testset, k=k, th=3.5)
    model_results["Item-Item CF"] = test_met_ii
    rmse_val = test_met_ii.get("RMSE", None)
    if rmse_val and rmse_val < best_rmse:
        best_rmse, best_model_name = rmse_val, "Item-Item CF"
    ok(f"Item-Item CF trained in {time.time()-t0:.1f}s | Test RMSE: {rmse_val:.4f}")
except Exception as e:
    print(f"  [ERROR] Item-Item CF failed: {e}")

# -- Model 4: SVD (Matrix Factorisation) -----------------------------------
print("\n  -- [4/5] SVD -- Matrix Factorisation --")
try:
    t0 = time.time()
    model_svd = CFRecommendationSystem(
        train_df, algo_class=SVD,
        params_grid={"n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.1},
    )
    model_svd.fit_svd(trainset)
    train_met_svd, test_met_svd = model_svd.evaluate(trainset, testset, k=k, th=3.5)
    model_results["SVD"] = test_met_svd
    rmse_val = test_met_svd.get("RMSE", None)
    if rmse_val and rmse_val < best_rmse:
        best_rmse, best_model_name = rmse_val, "SVD"
    ok(f"SVD trained in {time.time()-t0:.1f}s | Test RMSE: {rmse_val:.4f}")
except Exception as e:
    print(f"  [ERROR] SVD failed: {e}")

# -- Model 5: Hybrid -------------------------------------------------------
print("\n  -- [5/5] Hybrid (CF + Bayesian Rank) --")
try:
    t0 = time.time()
    model_rank_h = RankRecommendationSystem(method="bayesian")
    model_rank_h.compute_scores(train_df, gb_feature="prod_id", filter_feature="rating")
    model_cf_h = CFRecommendationSystem(
        train_df, algo_class=KNNBasic,
        params_grid={"k": 40, "sim_options": {"name": "cosine", "user_based": True}},
    )
    model_cf_h.fit_knn(trainset)
    model_hybrid = HybridRecommendationSystem(
        model_cf_h, model_rank_h, weight_cf=0.6, weight_rank=0.4
    )
    train_met_h, test_met_h = model_hybrid.evaluate(trainset, testset, k=k, th=3.5)
    model_results["Hybrid"] = test_met_h
    rmse_val = test_met_h.get("RMSE", None)
    if rmse_val and rmse_val < best_rmse:
        best_rmse, best_model_name = rmse_val, "Hybrid"
    ok(f"Hybrid trained in {time.time()-t0:.1f}s | Test RMSE: {rmse_val:.4f}")
except Exception as e:
    print(f"  [ERROR] Hybrid failed: {e}")

# ---------------------------------------------------------------------------
# STEP 6 -- MODEL COMPARISON TABLE
# ---------------------------------------------------------------------------
section("Model Comparison", step=6)

if model_results:
    results_df = pd.DataFrame(model_results).T.round(4)
    print(results_df.to_string())
    print(f"\n  >>> Best Model: {best_model_name}  (RMSE: {best_rmse:.4f}) <<<")
else:
    print("  No models were successfully trained.")

# ---------------------------------------------------------------------------
# STEP 7 -- SAVE BEST MODEL
# ---------------------------------------------------------------------------
section("Saving Best SVD Model", step=7)

models_dir = os.path.join(PROJECT_ROOT, "models")
os.makedirs(models_dir, exist_ok=True)
model_save_path = os.path.join(models_dir, "final_model_svd.pkl")

try:
    with open(model_save_path, "wb") as f:
        pickle.dump(
            {
                "model": model_svd.model,
                "train_df": train_df,
                "rating_scale": (df["rating"].min(), df["rating"].max()),
            },
            f,
        )
    ok(f"SVD model saved -> {model_save_path}")
except Exception as e:
    print(f"  [ERR]  Could not save model: {e}")

# ---------------------------------------------------------------------------
# STEP 8 -- SAMPLE RECOMMENDATIONS
# ---------------------------------------------------------------------------
section("Sample Recommendations", step=8)

try:
    # Top-5 products by Bayesian rating (Rank-Based)
    print("\n  Rank-Based -- Top 5 Products (Bayesian Average Rating):")
    top_prods = (
        model_rank.scores["bayesian_rating"]
        .nlargest(5)
        .round(3)
        .reset_index()
    )
    top_prods.columns = ["Product ID", "Bayesian Rating"]
    print(top_prods.to_string(index=False))
except Exception as e:
    print(f"  Could not generate rank-based sample: {e}")

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("  PIPELINE COMPLETE")
print("=" * 80)

print(f"\n  [OK] Dataset       : {n_interactions:,} interactions "
      f"({n_users:,} users x {n_products:,} products)")
print(f"  [OK] Models trained: {len(model_results)} / 5")
if best_model_name:
    print(f"  [OK] Best model    : {best_model_name}  (RMSE: {best_rmse:.4f})")
print(f"  [OK] Charts saved  : {figures_dir}")
print(f"  [OK] Model saved   : {model_save_path}")

# ---------------------------------------------------------------------------
# AUTO-DISPLAY CHARTS
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("  OPENING CHARTS")
print("=" * 80)

CHARTS = [
    ("rating_distribution.png",        "Rating Distribution"),
    ("reviews_per_user.png",           "Reviews per User"),
    ("reviews_per_product.png",        "Reviews per Product"),
    ("user_product_interactions.png",  "User & Product Interactions"),
    ("model_performance_comparison.png","Model Performance Comparison"),
]

available_charts = [
    (os.path.join(figures_dir, fn), title)
    for fn, title in CHARTS
    if os.path.isfile(os.path.join(figures_dir, fn))
]

if available_charts:
    import matplotlib.image as mpimg

    n = len(available_charts)
    cols = 2
    rows = (n + 1) // cols

    # Use a non-interactive backend-safe approach: save a combined figure
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 6))
    fig.suptitle(
        "Amazon Product Recommendation System  |  EDA & Model Results",
        fontsize=15, fontweight="bold", y=1.01
    )
    axes_flat = axes.flatten()

    for idx, (path, title) in enumerate(available_charts):
        img = mpimg.imread(path)
        axes_flat[idx].imshow(img)
        axes_flat[idx].set_title(title, fontsize=12, fontweight="bold", pad=8)
        axes_flat[idx].axis("off")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()

    combined_path = os.path.join(figures_dir, "all_charts_combined.png")
    plt.savefig(combined_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Combined chart saved -> {combined_path}")

    # Open with OS default viewer
    import platform, subprocess
    system = platform.system()
    print(f"  Opening combined chart in your default image viewer...")
    try:
        if system == "Windows":
            os.startfile(combined_path)
        elif system == "Darwin":
            subprocess.Popen(["open", combined_path])
        else:
            subprocess.Popen(["xdg-open", combined_path])
    except Exception as e:
        print(f"  Could not auto-open: {e}")
        print(f"  Manually open: {combined_path}")
else:
    print("  No charts to display.")

print("\n  Next steps:")
print("    python demo.py                          - instant demo with saved model")
print("    python demo.py --charts                 - view all charts in a window")
print("    python demo.py --charts-native          - open charts in Photos/Preview")
print("    python demo.py --user <user_id> --top 5 - personalised recommendations")
print("    jupyter notebook notebooks/product_recommendation_system.ipynb")
print()
